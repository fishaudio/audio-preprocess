from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import new
from pathlib import Path
from unicodedata import normalize

import epitran
import yaml
from g2pw import G2PWConverter
from loguru import logger
from tqdm import tqdm

# change pinyin labels based on polyphonic.yaml
polyponic_file = Path(__file__).parent / "polyphonic.yaml"
polyponic = yaml.safe_load(polyponic_file.open("r", encoding="utf-8"))

ch_ipa = epitran.Epitran("cmn-Latn")

chn_conv = G2PWConverter(style="pinyin", enable_non_tradional_chinese=True)

digit_dict = {
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "0": "零",
}


def digit_to_chinese(label):
    non_digit_label = ""
    for i in range(len(label)):
        non_digit_label += digit_dict.get(label[i], label[i])
    return non_digit_label


def label_list_to_str(label_list, phonetic=True):
    labels = ""
    tones_to_ipa = {"0": "˥", "1": "˦", "2": "˧", "3": "˨", "4": "˩", "5": "˥"}
    for label in label_list:
        if phonetic:
            print(label)
            l = ch_ipa.transliterate(label)
            l = normalize("NFC", l)
            new_l = ""
            for i in l:
                new_l += tones_to_ipa.get(i, i)
            labels += new_l + " "
        elif label != None and not phonetic:
            labels += label + " "
        else:
            labels += ""
    return labels


def modify_polyphonic(text, pinyin_label):
    for key, value in polyponic.items():
        key_len = len(key)
        for i in range(len(text) - key_len + 1):
            if text[i : i + key_len] == key:
                pinyin_label[i : i + key_len] = value
    # special case of 和 = han4 -> he2
    if "和" in text:
        if pinyin_label[text.index("和")] == "han4":
            pinyin_label[text.index("和")] = "he2"
    return pinyin_label


def g2p(file_loc, label):
    label = digit_to_chinese(label)
    pinyin_label = chn_conv(label)[0]
    pinyin_label = modify_polyphonic(label, pinyin_label)
    new_pinyin_label = []
    pinyin = ""
    phoneme = ""
    buf = ""
    for i in range(len(pinyin_label)):
        if pinyin_label[i] is not None:
            if buf != "":
                new_pinyin_label.append(buf)
                buf = ""
            new_pinyin_label.append(pinyin_label[i])
        else:
            buf += label[i]
    if buf != "":
        new_pinyin_label.append(buf)
    if new_pinyin_label[-1] == "\n":
        new_pinyin_label = new_pinyin_label[:-1]
    try:
        pinyin = label_list_to_str(new_pinyin_label, phonetic=False)
        phoneme = label_list_to_str(new_pinyin_label, phonetic=True)
    except:
        logger.error(
            f"Error in {file_loc} with label {label} and new pinyin_label {new_pinyin_label}"
        )
    return file_loc, phoneme, pinyin


def chinese2p(input_file, pinyin_labels_file, phoneme_labels_file, num_workers=8):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    file_loc, label = line.split("|")
                    tasks.append(executor.submit(g2p, file_loc, label))
                except:
                    logger.error(f"Error in {line}")
                    continue
        logger.info(f"Done submitting tasks, total: {len(tasks)}")

        with open(pinyin_labels_file, "w", encoding="utf-8") as f_pinyin, open(
            phoneme_labels_file, "w", encoding="utf-8"
        ) as f_phoneme:
            for task in tqdm(
                as_completed(tasks), total=len(tasks), desc="Converting to pinyin"
            ):
                file_loc, phoneme, pinyin_label = task.result()
                if (
                    pinyin_label == ""
                    or phoneme == ""
                    or pinyin_label == None
                    or phoneme == None
                    or pinyin_label == " "
                    or phoneme == " "
                ):
                    continue
                f_pinyin.write(file_loc + "|" + pinyin_label + "\n")
                f_phoneme.write(file_loc + "|" + phoneme + "\n")

    logger.info("Done")
