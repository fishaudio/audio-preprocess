from pathlib import Path
from typing import Literal

from loguru import logger
from tqdm import tqdm

PROMPT = {
    "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
    "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement.",
    "jp": "先進技術の領域において、人工知能の進化は画期的な成果として立っています。常に機械ができることの限界を押し広げているこのダイナミックな分野は、急速な成長と革新を見せています。複雑なデータパターンの解読から自動運転車の操縦まで、AIの応用は広範囲に及びます。",
}

ASRModelType = Literal["funasr", "whisper"]


def batch_transcribe(
    files: list[Path],
    model_size: str,
    model_type: ASRModelType,
    lang: str,
    pos: int,
):
    results = {}
    if model_type == "whisper":
        import whisper

        logger.info(f"Loading {model_size} model for {lang} transcription")
        model = whisper.load_model(model_size)
        for file in tqdm(files, position=pos):
            if lang in PROMPT:
                result = model.transcribe(
                    file, language=lang, initial_prompt=PROMPT[lang]
                )
            else:
                result = model.transcribe(file, language=lang)
            results[str(file)] = result["text"]
    elif model_type == "funasr":
        from funasr import AutoModel

        logger.info(f"Loading {model_size} model for {lang} transcription")
        model = AutoModel(
            model=model_size,
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            log_level="ERROR",
            disable_pbar=True,
        )
        for file in tqdm(files, position=pos):
            if lang in PROMPT:
                result = model.generate(
                    input=file, batch_size_s=300, hotword=PROMPT[lang]
                )
            else:
                result = model.generate(input=file, batch_size_s=300)
            # print(result)
            if isinstance(result, list):
                results[str(file)] = "".join([item["text"] for item in result])
            else:
                results[str(file)] = result["text"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return results
