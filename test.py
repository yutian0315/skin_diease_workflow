from dataSet import Datasets
import torch.utils.data as data
from train import Trainer
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--model_name', type=str, default="resnet", help="Model type")
    parser.add_argument('--Model_Path', type=str, default=r"./bset_finetuned_model.pth", help="Path to model")
    parser.add_argument('--test_dir', type=str, default=r"./test", help="Test dataset path")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
    parser.add_argument('--class_num', type=int, default=29, help="Number of classes")
    parser.add_argument('--gpu', default='0', type=str, help="GPU to use for testing")

    args = parser.parse_args()

    # Select GPU for testing
    device = f"cuda:{int(args.gpu)}"

    # Prepare test data loader
    dataSet_test = Datasets(train_dir=args.test_dir, val_dir=args.test_dir, mode="val")
    test_loader = data.DataLoader(dataSet_test, batch_size=1, shuffle=False, drop_last=True, num_workers=1)

    # Initialize the Trainer class for testing
    trainer = Trainer(
        data_loader=None,  # No training data loader
        val_loader=None,   # No validation data loader
        test_loader=test_loader,
        max_epoch=0,       # No need for epochs during testing
        save_path=None,    # No saving path required
        device=device,
        class_num=args.class_num,
        lr=0,              # No learning rate needed
        pretrained=False,  # Not training, so no pretrained option
        model_path=args.Model_Path,
        model_name=args.model_name
    )

    # Perform evaluation (testing)
    test_acc = trainer.evaluate(test_loader, args.Model_Path)
    print(f"\nTest Accuracy: {test_acc:.4f}%")
