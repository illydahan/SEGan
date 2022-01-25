import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from Generator import Generator
from Discriminator import Discriminator

from tqdm import tqdm
from SoundDataset import SoundDataset
from utils import evalutate_batch_performance



def train(sound_loader, gen, disc, gen_optim, disc_optim, writer, sound_data, device, g_lambda, epoch, best_results):
    best_pesq, best_ssnr = best_results
    
    
    running_disc_loss = 0
    running_gen_loss = 0
    loop = tqdm(sound_loader)
    for batch_idx, (clean_batch, noisy_batch) in enumerate(loop):
        # clean_batch = emphasis(clean_batch)
        # noisy_batch = emphasis(noisy_batch)
        
        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
        # Train D            
        # D real update
        
        disc.zero_grad()
    
        disc_real_out = disc(torch.cat([clean_batch.unsqueeze(1), noisy_batch.unsqueeze(1)], 1).to(device))
        clean_loss = torch.mean((disc_real_out - 1.0) ** 2)
        clean_loss.backward()
        
        
        # D fake update
        z = torch.normal(0, 1, ( noisy_batch.shape[0], 1024 , 8)).to(device)
        
        
        gen_fake_out = gen(noisy_batch.unsqueeze(1), z)
        
        fake_out = gen_fake_out.detach()

        disc_fake_out = disc(torch.cat([fake_out, noisy_batch.unsqueeze(1)], 1))
        
        noisy_loss = torch.mean((disc_fake_out) ** 2) 
        disc_loss = (clean_loss + noisy_loss) * 0.5
        
        
        noisy_loss.backward()
        disc_optim.step()
        
        # Train G
        gen.zero_grad()
        
        generated_output = gen(noisy_batch.unsqueeze(1), z)
        disc_fake_out = disc(torch.cat([generated_output, noisy_batch.unsqueeze(1)], 1))
        
        g_l1 = F.l1_loss(generated_output, clean_batch.unsqueeze(1))
        
        gen_loss =  torch.mean((disc_fake_out - 1) ** 2 ) +  g_l1*g_lambda
        
        #gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        
        running_disc_loss += disc_loss.item()
        running_gen_loss += gen_loss.item()
        
        
        if (batch_idx+1) % 50 == 0:
            
            with torch.no_grad():
                writer.add_scalar('Generator Loss', 
                                running_gen_loss / batch_idx,
                                epoch * len(sound_loader) + batch_idx+ 1)
                
                writer.add_scalar('Discriminator Loss', 
                                running_disc_loss / batch_idx,
                                epoch * len(sound_loader) + batch_idx + 1)
                
                
                writer.add_audio('Generated audio', 
                                generated_output[0, :] / ((2 << 14) - 1),
                                global_step=epoch * len(sound_loader) + batch_idx+ 1,
                                sample_rate=16e3)
                
                writer.add_audio('Noisy audio', 
                                noisy_batch[0, :] / ((2 << 14) - 1),
                                global_step=epoch * len(sound_loader) + batch_idx+ 1,
                                sample_rate=16e3)
                
                
                
                
                gen.eval()
                
                curr_ssnr, curr_pesq = evalutate_batch_performance(gen, sound_data ,sound_loader, device=device)
                
                    
                if curr_ssnr > best_ssnr:
                    #print(f"\nEpoch: {epoch+1} new best SSNR: {curr_ssnr:.3f}\n")
                    best_ssnr = curr_ssnr
                    
                    torch.save(gen.state_dict(), r'checkpoints/generator_parms_ssnr.pth')
                    torch.save(disc.state_dict(), r'checkpoints/disc_parms_ssnr.pth')
                    torch.save(gen_optim.state_dict(), r'checkpoints/gen_optim_ssnr.pth')
                    torch.save(disc_optim.state_dict(), r'checkpoints/disc_optim_ssnr.pth')
                    
                if curr_pesq > best_pesq:
                    #print(f"\nEpoch: {epoch+1} new best pesq: {curr_pesq:.3f}\n")
                    best_pesq = curr_pesq
                    
                    torch.save(gen.state_dict(), r'checkpoints/generator_parms_pesq.pth')
                    torch.save(disc.state_dict(), r'checkpoints/disc_parms_pesq.pth')
                    torch.save(gen_optim.state_dict(), r'checkpoints/gen_optim_pesq.pth')
                    torch.save(disc_optim.state_dict(), r'checkpoints/disc_optim_pesq.pth')
                    
                    
                gen.train()
                
            
                
                # disc_loss_history.append(running_disc_loss / 10)
                # gen_loss_history.append(running_gen_loss / 10)
                
                # running_gen_loss = 0
                # running_disc_loss = 0
                
            
        loop.set_postfix(gen_loss=gen_loss.item(), disc_loss = disc_loss.item(), ssnr=best_ssnr, pesq = best_pesq)
        
    #print(f" Epoch: {epoch} Batch: {batch_idx}: \nGenerator loss: {gen_loss_history[-1]:.3f}, Discriminator loss: {disc_loss_history[-1]:.3f},")
    
    return running_disc_loss / len(sound_loader), running_gen_loss / len(sound_loader), (best_pesq, best_ssnr)




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 100
    lr_gen = 1e-4
    lr_disc = 1e-4
    BATCH_SIZE = 32
    g_lambda = 100
    #mel_lambda = 500


    gen = Generator().to(device)
    disc = Discriminator().to(device)
    
    disc_loss_history = []
    gen_loss_history = []

    gen_optim = optim.Adam(gen.parameters(), lr_gen)
    disc_optim = optim.Adam(disc.parameters(), lr_disc)
    
    
    
    gen.load_state_dict(torch.load('checkpoints/generator_parms_ssnr.pth'))
    gen_optim.load_state_dict(torch.load('checkpoints/gen_optim_ssnr.pth'))
    
    disc.load_state_dict(torch.load('checkpoints/disc_parms_ssnr.pth'))
    disc_optim.load_state_dict(torch.load('checkpoints/disc_optim_ssnr.pth'))

    print("Loading dataset..")
    sound_ds = SoundDataset(r'sound_data/clean', r'sound_data/noisy')
    
    sound_loader = DataLoader(sound_ds, BATCH_SIZE, shuffle=True)
    print("Dataset loadded..")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("runs/segan")

    best_results = (0, 0)
    for epoch in range(n_epochs):
        disc_loss , gen_loss, best_results = train(sound_loader, gen, disc, gen_optim, disc_optim, writer, sound_ds, device, g_lambda, epoch, best_results=best_results)
        print(f"Epoch: {epoch+1} | Generator loss: {gen_loss:.3f} | Discriminator loss: {disc_loss:.3f}")
        