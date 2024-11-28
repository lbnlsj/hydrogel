import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import psutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class FiberHydrogelDataset(Dataset):
    def __init__(self, data_dir, mode='displacement', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.input_images = [f for f in os.listdir(os.path.join(data_dir, 'input'))
                             if f.endswith('_net.png')]

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_name = self.input_images[idx]
        index = input_name.split('_')[0]

        input_path = os.path.join(self.data_dir, 'input', input_name)
        input_img = Image.open(input_path).convert('RGB')

        if self.mode == 'displacement':
            output_name = f"{index}_U.png"
            output_dir = 'output'
        elif self.mode == 'strain':
            output_name = f"{index}_LE.png"
            output_dir = 'output(LE)'

        output_path = os.path.join(self.data_dir, output_dir, output_name)
        output_img = Image.open(output_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, stride=1, padding=1),  # 改为 3x3 kernel, padding=1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def get_optimal_batch_size(data_dir, device='cpu'):
    # 测试不同batch sizes
    batch_sizes = [2, 4, 8, 16, 32, 64]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FiberHydrogelDataset(data_dir, transform=transform)

    times = []
    memory_usage = []

    for batch_size in batch_sizes:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 测试一个mini-batch的处理时间
            start = time.time()
            for i, (real_A, real_B) in enumerate(dataloader):
                real_A = real_A.to(device)
                real_B = real_B.to(device)
                if i == 0:
                    break
            end = time.time()

            times.append(end - start)

            # 获取当前内存使用
            if device == 'cuda':
                memory = torch.cuda.memory_allocated() / 1e6  # MB
            else:
                memory = psutil.Process().memory_info().rss / 1e6  # MB
            memory_usage.append(memory)

        except RuntimeError as e:
            print(f'Batch size {batch_size} caused out of memory')
            break

    # 选择处理时间短且内存使用适中的batch size
    scores = [time * mem for time, mem in zip(times, memory_usage)]
    optimal_idx = np.argmin(scores)
    optimal_batch = batch_sizes[optimal_idx]

    return optimal_batch


def train_model(data_dir, mode='displacement', num_epochs=100, device='cpu'):
    # 获取最优batch size
    # batch_size = get_optimal_batch_size(data_dir, device)
    # print(f'Using optimal batch size: {batch_size}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FiberHydrogelDataset(data_dir, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                             lr=0.0002, betas=(0.5, 0.999))

    losses_G = []
    losses_D = []

    for epoch in range(num_epochs):
        running_loss_G = 0.0
        running_loss_D = 0.0

        for i, (real_A, real_B) in enumerate(tqdm(dataloader)):
            batch_size = real_A.size(0)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            valid = torch.ones(batch_size, 1, 16, 16).to(device)
            fake = torch.zeros(batch_size, 1, 16, 16).to(device)

            # Train Generator
            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_G = (loss_GAN_AB + loss_GAN_BA +
                      (loss_cycle_A + loss_cycle_B) * 10.0 +
                      (loss_id_A + loss_id_B) * 5.0)

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            loss_real_A = criterion_GAN(D_A(real_A), valid)
            loss_real_B = criterion_GAN(D_B(real_B), valid)

            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)

            loss_D = (loss_real_A + loss_fake_A) / 2 + (loss_real_B + loss_fake_B) / 2

            loss_D.backward()
            optimizer_D.step()

            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()

        avg_loss_G = running_loss_G / len(dataloader)
        avg_loss_D = running_loss_D / len(dataloader)
        losses_G.append(avg_loss_G)
        losses_D.append(avg_loss_D)

        print(f"Epoch [{epoch}/{num_epochs}] Loss_D: {avg_loss_D:.4f} Loss_G: {avg_loss_G:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'losses_G': losses_G,
                'losses_D': losses_D
            }, f'checkpoint_{mode}_epoch_{epoch + 1}.pth')

            plt.figure(figsize=(10, 5))
            plt.plot(losses_G, label='Generator Loss')
            plt.plot(losses_D, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{mode.capitalize()} Model Training Losses')
            plt.legend()
            plt.savefig(f'losses_{mode}_epoch_{epoch + 1}.png')
            plt.close()

    return G_AB, G_BA, D_A, D_B, losses_G, losses_D


def test_model(model_path, test_image_path, device='cpu', mode='displacement'):
    checkpoint = torch.load(model_path, map_location=device)
    G_AB = Generator().to(device)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_AB.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_img = Image.open(test_image_path).convert('RGB')
    test_tensor = transform(test_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = G_AB(test_tensor)

    output = output.squeeze(0).cpu()
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output)

    output_filename = os.path.basename(test_image_path).replace('_net.png', f'_predicted_{mode}.png')
    output.save(output_filename)
    return output


if __name__ == "__main__":
    data_dir = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 100

    print("Training displacement model...")
    G_AB_disp, G_BA_disp, D_A_disp, D_B_disp, losses_G_disp, losses_D_disp = train_model(
        data_dir, mode='displacement', num_epochs=num_epochs, device=device)

    print("Training strain model...")
    G_AB_strain, G_BA_strain, D_A_strain, D_B_strain, losses_G_strain, losses_D_strain = train_model(
        data_dir, mode='strain', num_epochs=num_epochs, device=device)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_G_disp, label='Generator Loss')
    plt.plot(losses_D_disp, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Displacement Model Training Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses_G_strain, label='Generator Loss')
    plt.plot(losses_D_strain, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Strain Model Training Losses')
    plt.legend()

    plt.tight_layout()
    plt.savefig('final_training_losses.png')
    plt.close()
