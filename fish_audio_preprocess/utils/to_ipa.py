from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import new
from pathlib import Path
from unicodedata import normalize

import yaml
from loguru import logger
from tqdm import tqdm

# change pinyin labels based on polyphonic.yaml
polyponic_file = Path(__file__).parent / "polyphonic.yaml"

polyponic = None
ch_ipa = None
chn_conv = None

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
    return "".join(digit_dict.get(label[i], label[i]) for i in range(len(label)))


def label_list_to_str(label_list, phonetic=True):
    global ch_ipa

    if ch_ipa is None:
        import epitran

        ch_ipa = epitran.Epitran("cmn-Latn")

    labels = ""
    tones_to_ipa = {"0": "˥", "1": "˦", "2": "˧", "3": "˨", "4": "˩", "5": "˥"}
    for label in label_list:
        if phonetic:
            print(label)
            l = ch_ipa.transliterate(label)
            l = normalize("NFC", l)
            new_l = "".join(tones_to_ipa.get(i, i) for i in l)
            labels += f"{new_l} "
        elif label != None:
            labels += f"{label} "
        else:
            labels += ""
    return labels


def modify_polyphonic(text, pinyin_label):
    global polyponic
    if polyponic is None:
        with open(polyponic_file, "r", encoding="utf-8") as f:
            polyponic = yaml.safe_load(f)

    for key, value in polyponic.items():
        key_len = len(key)
        for i in range(len(text) - key_len + 1):
            if text[i : i + key_len] == key:
                pinyin_label[i : i + key_len] = value
    # special case of 和 = han4 -> he2
    if "和" in text and pinyin_label[text.index("和")] == "han4":
        pinyin_label[text.index("和")] = "he2"
    return pinyin_label


def g2p(file_loc, label):
    global chn_conv
    if chn_conv is None:
        import g2pw

        chn_conv = g2pw.G2PWConverter(style="pinyin", enable_non_tradional_chinese=True)

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
    except Exception:
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
                except Exception:
                    logger.error(f"Error in {line}")
                    continue
        logger.info(f"Done submitting tasks, total: {len(tasks)}")

        with (
            open(pinyin_labels_file, "w", encoding="utf-8") as f_pinyin,
            open(phoneme_labels_file, "w", encoding="utf-8") as f_phoneme,
        ):
            for task in tqdm(
                as_completed(tasks), total=len(tasks), desc="Converting to pinyin"
            ):
                file_loc, phoneme, pinyin_label = task.result()
                if (
                    pinyin_label == ""
                    or phoneme == ""
                    or pinyin_label is None
                    or phoneme is None
                    or pinyin_label == " "
                    or phoneme == " "
                ):
                    continue
                f_pinyin.write(f"{file_loc}|{pinyin_label}" + "\n")
                f_phoneme.write(f"{file_loc}|{phoneme}" + "\n")

    logger.info("Done")
