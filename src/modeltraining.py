import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy
from tqdm import tqdm
from utils.utils import encode_segmap, n_classes, visualize_sample
from data_processing.dataprocessing import get_dataloaders

class OurModel(LightningModule):
    def __init__(self, lr=1e-3, batch_size=16, num_workers=4):
        super(OurModel, self).__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.layer = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
        
        # Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.iou_metric = MulticlassJaccardIndex(num_classes=n_classes)
        
    def process(self, image, segment):
        out = self(image)
        segment = encode_segmap(segment)
        loss = self.criterion(out, segment.long())
        iou = self.iou_metric(out, segment)
        return loss, iou
    
    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_loader, _, _ = get_dataloaders(batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def val_dataloader(self):
        _, val_loader, _ = get_dataloaders(batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

def train_model(model, max_epochs=200):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='epoch-{epoch:02d}-val_loss-{val_loss:.2f}',
        save_top_k=-1,
        every_n_epochs=5,
        save_last=True
    )
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices='auto',
        precision=16,
        limit_val_batches=0.1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )
    
    trainer.fit(model)
    
    # Save model
    torch.save(model.state_dict(), "new_model_200EPOCHS_weights.pth")
    torch.save(model, "new_full_model_200_EPOCHS.pth")

def evaluate_model(model, data_dir='/kaggle/input/cityscapes/cityscapes', batch_size=8):
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    model.eval()
    model = model.cuda()
    
    iou_metric = MulticlassJaccardIndex(num_classes=n_classes).to('cuda')
    dice_metric = MulticlassF1Score(num_classes=n_classes, average='macro').to('cuda')
    pixel_acc_metric = MulticlassAccuracy(num_classes=n_classes, average='micro').to('cuda')
    
    all_ious, all_dices, all_pixel_accs = [], [], []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.cuda()
            masks = encode_segmap(masks).long().cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            iou = iou_metric(preds, masks)
            dice = dice_metric(preds, masks)
            pixel_acc = pixel_acc_metric(preds, masks)
            
            all_ious.append(iou.item())
            all_dices.append(dice.item())
            all_pixel_accs.append(pixel_acc.item())
    
    print(f"Validation Mean IoU: {sum(all_ious)/len(all_ious):.4f}")
    print(f"Validation Mean Dice: {sum(all_dices)/len(all_dices):.4f}")
    print(f"Validation Mean Pixel Accuracy: {sum(all_pixel_accs)/len(all_pixel_accs):.4f}")

def infer_model(model, test_loader, sample_idx=7):
    model.eval()
    model = model.cuda()
    
    with torch.no_grad():
        for batch in test_loader:
            img, seg = batch
            output = model(img.cuda())
            break
    
    encoded_mask = encode_segmap(seg[sample_idx].clone())
    visualize_sample(img, encoded_mask, output, sample=sample_idx)

if __name__ == "__main__":
    # Example usage
    model = OurModel()
    train_model(model)
    evaluate_model(model)
    _, _, test_loader = get_dataloaders()
    infer_model(model, test_loader)