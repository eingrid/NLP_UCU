import nlpaug.augmenter.word as naw
import torch
import pandas as pd
import re
from tqdm import tqdm
aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-uk-en',
    to_model_name='Helsinki-NLP/opus-mt-en-uk',
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def augment_text(text):
    #remove punct
    text = re.sub(r'[^\w\s]', '', text)
    return aug.augment(text)

text = "Як були черги величезні та багатогодинні, так і залишилися Ганьба Електронна хвалена черга приймає запис із мого питання 20 жовтня на 7 листопада найближча дата У результаті просто пекло  я 242 у черзі, людей у черзі з мого ж питання…"
aug_text = augment_text(text)
print(aug_text)

train_df = pd.read_csv("emotions/train.csv")

#define which data to augment
minority_classes = ['Sadness', 'Disgust', 'Fear', 'Surprise']

# Fix: Use train_test_split instead of sample with stratify
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df["emotion"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
val_df.to_csv("data/val.csv", index=False)
train_df.to_csv("data/train.csv", index=False)

#augment minority classes
augmented_data = []
for cls in minority_classes:
    cls_data = train_df[train_df['emotion'] == cls]
    #augment all samples
    for _, row in tqdm(cls_data.iterrows()):
        augmented_text = augment_text(row['text'])[0]
        augmented_data.append({'text': augmented_text, 'emotion': row['emotion'], 'category':row['category'], 'augmented': True})

#save val and augmented train data
augmented_df = pd.DataFrame(augmented_data)
train_df = pd.concat([train_df, augmented_df]).reset_index(drop=True)
train_df.to_csv("data/train_balanced_augmented.csv", index=False)