import torch
import torch.nn as nn
import pickle
import pandas as pd
import re
from difflib import SequenceMatcher

# === Токенізація, числення, декодування ===
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", "", text)
    return text.strip().split()

def numericalize(sentence, vocab):
    tokens = simple_tokenize(sentence)
    return [vocab['<sos>']] + [vocab.get(t, vocab['<unk>']) for t in tokens] + [vocab['<eos>']]

def decode_tokens(tokens, vocab):
    id2word = {idx: word for word, idx in vocab.items()}
    words = [id2word.get(t, '<unk>') for t in tokens]
    return ' '.join(words)

def similarity_percent(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    return round(SequenceMatcher(None, a, b).ratio(), 4)  # returns 0.0 to 1.0

# === Моделі ===
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

    def translate(self, src_tensor, trg_vocab, max_len=50):
        self.eval()
        device = src_tensor.device
        hidden = self.encoder(src_tensor.unsqueeze(0))
        input_token = torch.tensor([trg_vocab['<sos>']], device=device)
        translated_tokens = []

        for _ in range(max_len):
            output, hidden = self.decoder(input_token, hidden)
            top1 = output.argmax(1).item()
            if top1 == trg_vocab['<eos>']:
                break
            translated_tokens.append(top1)
            input_token = torch.tensor([top1], device=device)

        return translated_tokens

# === Завантаження словників ===
with open("vocab (2).pkl", "rb") as f:
    SRC_vocab, TRG_vocab = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM, OUTPUT_DIM = len(SRC_vocab), len(TRG_vocab)
EMB_DIM, HID_DIM = 128, 256

# === Завантаження моделей (будь-яка кількість) ===
model_paths = {
    'v1': 'model_epoch_1_step1000.pt',
    'v2': 'model_epoch_1_step5000.pt',
    'v3': 'model_epoch_1_step10000.pt',
    'v4': 'model_epoch_1_step20000.pt',
    'v5': 'model_epoch_1_step26800.pt',
    'v6': 'model_epoch_1_step300000.pt'
}
models = {}

for key, path in model_paths.items():
    model = Seq2Seq(
        Encoder(INPUT_DIM, EMB_DIM, HID_DIM),
        Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models[key] = model

# === Функція для обробки одного CSV ===
def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    for i, row in df.iterrows():
        word = str(row["word"])
        expected = str(row["expectation"])

        for version, model in models.items():
            try:
                src_numerical = numericalize(word, SRC_vocab)
                src_tensor = torch.tensor(src_numerical, dtype=torch.long).to(device)

                with torch.no_grad():
                    output_ids = model.translate(src_tensor, TRG_vocab)
                    translated = decode_tokens(output_ids, TRG_vocab)
            except Exception as e:
                translated = f"<error: {e}>"

            sim = similarity_percent(translated, expected)
            df.at[i, f"result_{version}"] = translated
            df.at[i, f"present_{version}"] = sim

    # === Впорядкування колонок
    base_columns = ['word', 'expectation']
    result_columns = sorted([col for col in df.columns if col.startswith('result_')], key=lambda x: int(x.split('_v')[-1]))
    present_columns = sorted([col for col in df.columns if col.startswith('present_')], key=lambda x: int(x.split('_v')[-1]))
    ordered_columns = base_columns + result_columns + present_columns
    df = df[ordered_columns]

    df.to_csv(output_file, index=False)
    print(f"✅ Файл оброблено: {output_file}")


# === Обробка CSV-файлів ===
process_csv("test_case_easy.csv", "test_case_easy_output.csv")
process_csv("test_case_hard.csv", "test_case_hard_output.csv")
