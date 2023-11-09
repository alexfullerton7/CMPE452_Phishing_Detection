import CNN as net
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import datetime
import matplotlib.pyplot as plt
import custom_dataset as dataset
from torch.optim.lr_scheduler import ExponentialLR


def train(n_epochs, batch_size, gamma, optimizer, scheduler, model, content_set, style_set, device, save_model_path,
          save_plot_path):
    # Load datasets in with Dataloader according to the batch size specified
    content_loader = DataLoader(content_set, batch_size=batch_size, shuffle=True)
    style_loader = DataLoader(style_set, batch_size=batch_size, shuffle=True)

    print("training...")
    model.train()  # Keep track of gradient for backtracking

    # Track losses every epoch
    losses_train = []
    losses_content = []
    losses_style = []
    # Track epochs for plotting
    epochs = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch)

        # Initialize a new list for this epoch   
        loss_train = 0.00
        loss_c_train = 0.00
        loss_s_train = 0.00

        content_iter = iter(content_loader)
        style_iter = iter(style_loader)

        num_batches = int(len(content_set) / batch_size)
        
        # Iterate through content and style batches together
        for batch in range(num_batches):
            # for content_images, style_images in zip(content_loader, style_loader):
            content_images = next(content_iter).to(device=device)
            style_images = next(style_iter).to(device=device)

            # Forward pass through model to get content and style loss
            loss_c, loss_s = model(content_images, style_images)
            loss_s = loss_s * gamma

            # Total loss is a combination of content and style loss
            total_loss = loss_c + loss_s
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_train += (loss_c.item() + loss_s.item())
            loss_c_train += loss_c.item()
            loss_s_train += loss_s.item()

        scheduler.step()

        # Calculate the average loss over batches for the entire epoch
        losses_train += [loss_train / len(content_loader)]
        losses_content += [loss_c_train / len(content_loader)]
        losses_style += [loss_s_train / len(content_loader)]

        # Arrays for plotting loss
        epochs.append(epoch)
        print('{} Epoch {}, Style loss {}, Content loss {}, Total loss {}'.format(datetime.datetime.now(), epoch, loss_s.item(), loss_c.item(), total_loss.item()))

    # Save decoder weights for inference
    torch.save(model.decoder.state_dict(), save_model_path)

    # Plot training loss over epochs
    plt.plot(epochs, losses_train, label='Total Loss', color='blue')
    plt.plot(epochs, losses_content, label='Content Loss', color='green')
    plt.plot(epochs, losses_style, label='Style Loss', color='red')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(save_plot_path)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # Set up terminal inputs
    parser.add_argument('-content_dir', '--content-dir', type=str, default='./Data/COCO1k/')
    parser.add_argument('-style_dir', '--style-dir', type=str, default='./Data/wikiart1k/')
    parser.add_argument('-gamma', '--gamma', type=float, default=0.9)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-l', '--encoder-dir', type=str, default='./encoder.pth')
    parser.add_argument('-s', '--save-model', type=str, default='decoder.pth')
    parser.add_argument('-p', '--save-plot', type=str, default='loss.NST.png')
    parser.add_argument('-cuda', '--device', type=str, default='Y')

    args = parser.parse_args()

    # Assign terminal inputs to appropriate variables
    content_path = args.content_dir
    style_path = args.style_dir
    gamma = args.gamma
    n_epochs = args.epochs
    batch_size = args.batch_size
    encoder_path = args.encoder_dir
    save_model_path = args.save_model
    save_plot_path = args.save_plot

    encoder = net.encoder_decoder.encoder
    encoder_weights = torch.load(encoder_path)
    encoder.load_state_dict(encoder_weights)

    # Creates an instance of the class and runs __init__
    model = net.AdaIN_net(encoder=encoder, decoder=net.encoder_decoder.decoder)

    # Set device to Cuda if available
    device = "cpu"
    if args.device == "Y":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Cuda or cpu?: {device}")
        # Move model to the GPU if available
        model.to(device)

    # Applies transform to normalize all the images for training
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    # Create an instance of the content dataset
    content_set = dataset.custom_dataset(content_path, transform=transform)

    # Create an instance of the style dataset
    style_set = dataset.custom_dataset(style_path, transform=transform)

    # Use the Adam optimizer
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)
    # Create scheduler
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.95)

    # Calling the train method
    train(n_epochs, batch_size, gamma, optimizer, scheduler, model, content_set, style_set, device, save_model_path,
          save_plot_path)


if __name__ == "__main__":
    main()
