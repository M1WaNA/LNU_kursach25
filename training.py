import os
import re
import time
import random
import pickle
import threading
import platform
import psutil
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


# === Логування ===
class TkinterLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.after(0, self.text_widget_insert, msg)

    def text_widget_insert(self, msg):
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)


# === 1. Токенізатор ===
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", "", text)
    return text.strip().split()


# === 2. Словник ===
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(simple_tokenize(sent))
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def numericalize(sentence, vocab):
    tokens = simple_tokenize(sentence)
    return [vocab['<sos>']] + [vocab.get(t, vocab['<unk>']) for t in tokens] + [vocab['<eos>']]


# === 3. Dataset ===
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        self.src = [numericalize(s, src_vocab) for s in src_sentences]
        self.trg = [numericalize(s, trg_vocab) for s in trg_sentences]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.trg[idx], dtype=torch.long)


# === 4. Модель ===
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        _, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(src.device)
        hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs


class TrainingArguments:
    def __init__(
        self,
        output_dir="./output",
        dataloader_pin_memory=True,
        per_device_train_batch_size=32,
        num_train_epochs=1,
        save_steps=1000,
        save_total_limit=3,
        logging_dir="./logs",
        logging_steps=1000
    ):
        self.output_dir = output_dir
        self.dataloader_pin_memory = dataloader_pin_memory
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps


# === 5. GUI з тренуванням ===
class TrainerApp:
    def __init__(self, root):
        self.root = root
        root.title("Тренування перекладача")
        self.file_path = None
        self.is_training = False
        self.pause_flag = False
        self.stop_flag = False

        # === UI ===
        self.label_file = tk.Label(root, text="Файл не обрано")
        self.label_file.pack(pady=5)

        self.btn_browse = tk.Button(root, text="Вибрати файл", command=self.browse_file)
        self.btn_browse.pack()

        self.btn_start = tk.Button(root, text="Старт", command=self.start_training, state="disabled")
        self.btn_start.pack()

        self.btn_pause = tk.Button(root, text="Пауза", command=self.pause_training, state="disabled")
        self.btn_pause.pack()

        self.btn_stop = tk.Button(root, text="Стоп", command=self.stop_training, state="disabled")
        self.btn_stop.pack()

        self.progress_label = tk.Label(root, text="Прогрес: 0%")
        self.progress_label.pack()

        self.epoch_label = tk.Label(root, text="Епоха: 0 / 0")
        self.epoch_label.pack()

        self.batch_label = tk.Label(root, text="Батч: 0 / 0")
        self.batch_label.pack()

        self.remaining_time_label = tk.Label(root, text="Час до завершення епохи: ...")
        self.remaining_time_label.pack(pady=5)

        self.console = scrolledtext.ScrolledText(root, height=20, width=100)
        self.console.pack()

        # === Логування в файл + GUI ===
        self.logger = logging.getLogger("TrainerLogger")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("train_log.txt")
        gui_handler = TkinterLogHandler(self.console)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        gui_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(gui_handler)

        self.log_system_info()

    def log_system_info(self):
        info = [
            f"Система: {platform.system()} {platform.release()}",
            f"Процесор: {platform.processor()}",
            f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB",
            f"CUDA доступна: {torch.cuda.is_available()}",
            f"CUDA пристрій: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        ]
        for line in info:
            self.logger.info(line)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Всі файли", "*.*")])
        if path:
            self.file_path = path
            self.label_file.config(text=f"Вибрано: {os.path.basename(path)}")
            self.btn_start.config(state="normal")
            self.logger.info(f"Файл обрано: {path}")

    def start_training(self):
        if self.is_training:
            return
        self.logger.info("Старт тренування")

        # Приклад: ініціалізація аргументів тренування
        self.training_args = TrainingArguments(
            output_dir="./lora_croissantllm",
            dataloader_pin_memory=False,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            save_steps=1000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=1000,
        )

        self.is_training = True
        self.pause_flag = False
        self.stop_flag = False
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="Пауза")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self.train_model).start()


    def pause_training(self):
        if self.is_training:
            self.pause_flag = not self.pause_flag
            state = "Пауза" if not self.pause_flag else "Продовжити"
            self.logger.info(f"{state} натиснута")
            self.btn_pause.config(text=state)

    def stop_training(self):
        if self.is_training:
            self.logger.info("Зупинка тренування")
            self.stop_flag = True

    def update_progress(self, percent):
        self.root.after(0, lambda: self.progress_label.config(text=f"Прогрес: {percent}%"))

    def update_epoch(self, current, total):
        self.root.after(0, lambda: self.epoch_label.config(text=f"Епоха: {current} / {total}"))

    def update_batch(self, current, total):
        self.root.after(0, lambda: self.batch_label.config(text=f"Батч: {current} / {total}"))


    def save_model_and_vocab(self, model, SRC_vocab, TRG_vocab, epoch):
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
        with open("vocab.pkl", "wb") as f:
            pickle.dump((SRC_vocab, TRG_vocab), f)
        self.logger.info(f"Збережено модель після епохи {epoch}")

    def train_model(self):
        df = pd.read_csv(self.file_path, sep="|", header=None, on_bad_lines='skip')
        src_sentences = df[0].fillna("").tolist()
        trg_sentences = df[1].fillna("").tolist()

        SRC_vocab = build_vocab(src_sentences)
        TRG_vocab = build_vocab(trg_sentences)

        split_idx = int(0.9 * len(src_sentences))
        train_dataset = TranslationDataset(src_sentences[:split_idx], trg_sentences[:split_idx], SRC_vocab, TRG_vocab)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        INPUT_DIM = len(SRC_vocab)
        OUTPUT_DIM = len(TRG_vocab)
        EMB_DIM, HID_DIM = 128, 256

        model = Seq2Seq(
            Encoder(INPUT_DIM, EMB_DIM, HID_DIM),
            Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
        ).to(device)

        optimizer = optim.Adam(model.parameters())
        PAD_IDX = SRC_vocab['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        def collate_fn(batch):
            src_batch, trg_batch = zip(*batch)
            src = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
            trg = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
            return src.to(device), trg.to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=self.training_args.dataloader_pin_memory
        )
        N_EPOCHS = self.training_args.num_train_epochs

        total_steps = len(train_loader)
        current_step = 0

        for epoch in range(1, N_EPOCHS + 1):
            if self.stop_flag:
                self.save_model_and_vocab(model, SRC_vocab, TRG_vocab, epoch - 1)
                break

            self.update_epoch(epoch, N_EPOCHS)
            model.train()
            epoch_loss = 0
            start_time = time.time()

            total_batches = len(train_loader)
            for i, (src, trg) in enumerate(train_loader, 1):
                self.logger.info(f"Епоха {epoch}, батч {i}/{total_batches} почався")
                while self.pause_flag:
                    time.sleep(0.1)
                    if self.stop_flag:
                        break
                if self.stop_flag:
                    self.save_model_and_vocab(model, SRC_vocab, TRG_vocab, epoch)
                    self.reset_buttons()
                    return

                batch_start = time.time()

                optimizer.zero_grad()
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                current_step += 1
                self.update_progress(int((current_step / total_steps) * 100))
                self.update_batch(i, total_batches)

                elapsed = time.time() - start_time
                if current_step > 0:
                    estimated_total = elapsed / current_step * total_steps
                    remaining = estimated_total - elapsed
                else:
                    remaining = 0
                    
                days = int(remaining) // 86400
                hours = (int(remaining) % 86400) // 3600
                minutes = (int(remaining) % 3600) // 60
                seconds = int(remaining) % 60

                time_str = f"{days}д {hours}г {minutes}хв {seconds}с"
                self.root.after(0, lambda t=time_str: self.remaining_time_label.config(text=f"Час до завершення: {t}"))

                if i % self.training_args.logging_steps == 0:
                    self.logger.info(f"[Лог] Епоха {epoch}, Батч {i}, Втрата: {batch_loss:.4f}")

                if i % self.training_args.save_steps == 0:
                    self.save_model_and_vocab(model, SRC_vocab, TRG_vocab, f"{epoch}_step{i}")

                torch.cuda.empty_cache()

            self.save_model_and_vocab(model, SRC_vocab, TRG_vocab, epoch)

        self.reset_buttons()
        self.update_progress(0)
        self.update_epoch(0, 0)
        self.update_batch(0, 0)
        self.root.after(0, lambda: self.remaining_time_label.config(text="Час до завершення: ---"))



    def reset_buttons(self):
        self.is_training = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled", text="Пауза")
        self.btn_stop.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerApp(root)
    root.mainloop()
