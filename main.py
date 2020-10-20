from train import Trainer

def main():
    index = 0
    setting_csv_path = "./setting.csv"
    trainer = Trainer(setting_csv_path=setting_csv_path, index=index)
    trainer.train()

if __name__ == "__main__":
    main()
