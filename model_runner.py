import sys
import json
import os
import tempfile
import traceback
import torch

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _best_device()

TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_stt_cache: dict = {}
_tts_cache: dict = {}

DEFAULT_STT_MODEL = "openai/whisper-tiny"

def _load_stt(model_id: str):
    
    if model_id not in _stt_cache:
        from transformers import pipeline

        print(json.dumps({"status": "loading", "task": "stt", "model": model_id}),
              flush=True)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        _stt_cache[model_id] = pipe

    return _stt_cache[model_id]

def run_stt(model_id: str, audio_path: str, params: dict) -> dict:
    
    if not os.path.exists(audio_path):
        return {"status": "error", "error": f"Audio file not found: {audio_path}"}

    pipe = _load_stt(model_id)

    generate_kwargs: dict = {}
    if params.get("language"):
        generate_kwargs["language"] = params["language"]
        generate_kwargs["task"] = "transcribe"

    result = pipe(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=params.get("return_timestamps", False),
        chunk_length_s=params.get("chunk_length_s", 30),
        batch_size=params.get("batch_size", 8),
    )

    out: dict = {"status": "success", "text": result["text"].strip(), "model": model_id}
    if params.get("return_timestamps") and "chunks" in result:
        out["chunks"] = result["chunks"]
    return out

DEFAULT_TTS_MODEL = "facebook/mms-tts-eng"

def _load_tts_mms(model_id: str) -> dict:
    
    if model_id not in _tts_cache:
        from transformers import VitsModel, AutoTokenizer

        print(json.dumps({"status": "loading", "task": "tts", "model": model_id}),
              flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = VitsModel.from_pretrained(model_id).to(DEVICE)

        if DEVICE == "cuda":
            model = model.half()

        _tts_cache[model_id] = {
            "kind":        "mms",
            "tokenizer":   tokenizer,
            "model":       model,
            "sample_rate": model.config.sampling_rate,
        }

    return _tts_cache[model_id]

def _load_tts_speecht5(model_id: str) -> dict:
    
    if model_id not in _tts_cache:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        print(json.dumps({"status": "loading", "task": "tts", "model": model_id}),
              flush=True)

        processor = SpeechT5Processor.from_pretrained(model_id)
        model     = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(DEVICE)
        vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

        ds          = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_emb = torch.tensor(ds[7306]["xvector"]).unsqueeze(0).to(DEVICE)

        _tts_cache[model_id] = {
            "kind":        "speecht5",
            "processor":   processor,
            "model":       model,
            "vocoder":     vocoder,
            "speaker_emb": speaker_emb,
            "sample_rate": 16000,
        }

    return _tts_cache[model_id]

def _detect_tts_family(model_id: str) -> str:
    mid = model_id.lower()
    if "mms-tts" in mid or "vits" in mid:
        return "mms"
    if "speecht5" in mid:
        return "speecht5"

    return "mms"

def run_tts(model_id: str, text: str, params: dict) -> dict:
    
    import soundfile as sf

    family = _detect_tts_family(model_id)

    if family == "mms":
        ctx    = _load_tts_mms(model_id)
        inputs = ctx["tokenizer"](text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = ctx["model"](**inputs)
        audio = output.waveform[0].cpu().float().numpy()

    elif family == "speecht5":
        ctx = _load_tts_speecht5(model_id)
        spk = ctx["speaker_emb"]

        if params.get("speaker_idx") is not None:
            from datasets import load_dataset
            ds  = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            spk = torch.tensor(
                ds[int(params["speaker_idx"])]["xvector"]
            ).unsqueeze(0).to(DEVICE)

        inputs = ctx["processor"](text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            speech = ctx["model"].generate_speech(
                inputs["input_ids"], spk, vocoder=ctx["vocoder"]
            )
        audio = speech.cpu().float().numpy()

    else:
        return {"status": "error", "error": f"Unrecognised TTS family for model: {model_id}"}

    output_path = params.get("output_path") or os.path.join(
        tempfile.mkdtemp(), "tts_output.wav"
    )
    sf.write(output_path, audio, ctx["sample_rate"])

    return {
        "status":      "success",
        "file":        output_path,
        "sample_rate": ctx["sample_rate"],
        "model":       model_id,
    }

def dispatch(request: dict) -> dict:
    task   = request.get("task", "")
    model  = request.get("model", "")
    inp    = request.get("input", "")
    params = request.get("params", {})

    if task == "stt":
        return run_stt(model or DEFAULT_STT_MODEL, inp, params)

    elif task == "tts":
        return run_tts(model or DEFAULT_TTS_MODEL, inp, params)

    elif task == "info":
        return {
            "status":      "success",
            "device":      DEVICE,
            "torch_dtype": str(TORCH_DTYPE),
            "stt_default": DEFAULT_STT_MODEL,
            "tts_default": DEFAULT_TTS_MODEL,
            "loaded_stt":  list(_stt_cache.keys()),
            "loaded_tts":  list(_tts_cache.keys()),
        }

    else:
        return {
            "status": "error",
            "error":  f"Unknown task '{task}'. Valid tasks: stt, tts, info.",
        }

def main():
    print(json.dumps({"status": "ready", "device": DEVICE}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"status": "error", "error": f"JSON parse error: {exc}"}),
                  flush=True)
            continue

        try:
            result = dispatch(request)
        except Exception as exc:
            result = {
                "status":    "error",
                "error":     str(exc),
                "traceback": traceback.format_exc(),
            }

        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
