import torch
import torch.nn as nn
import pickle
import re

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

# === Завантаження моделі та словників ===
with open("vocab (2).pkl", "rb") as f:
    SRC_vocab, TRG_vocab = pickle.load(f)

INPUT_DIM = len(SRC_vocab)
OUTPUT_DIM = len(TRG_vocab)
EMB_DIM, HID_DIM = 128, 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(
    Encoder(INPUT_DIM, EMB_DIM, HID_DIM),
    Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
).to(device)

model.load_state_dict(torch.load("model_epoch_1_step300000.pt", map_location=device))
model.eval()

# === Цикл перекладу без виходу ===
print("Введіть речення для перекладу (Ctrl+C щоб завершити):")
while True:
    try:
        sentence = input("> ")
        src_numerical = numericalize(sentence, SRC_vocab)
        src_tensor = torch.tensor(src_numerical, dtype=torch.long).to(device)

        with torch.no_grad():
            translated_token_ids = model.translate(src_tensor, TRG_vocab)
            translated_text = decode_tokens(translated_token_ids, TRG_vocab)
            print(f"Переклад: {translated_text}")
    except KeyboardInterrupt:
        print("\nПереклад завершено вручну.")
        break
    except Exception as e:
        print(f"Помилка: {e}")
