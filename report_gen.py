import os, json, glob, re
import matplotlib.pyplot as plt

RUNS = [
    ("nowarm",    "cz-en/checkpoints_nowarm"),
    ("warmconst", "cz-en/checkpoints_warmconst"),
    ("warmlin",   "cz-en/checkpoints_warmlin"),
]
BLEU_LOGS = {
    "nowarm":    "cz-en/logs/bleu_nowarm.txt",
    "warmconst": "cz-en/logs/bleu_warmconst.txt",
    "warmlin":   "cz-en/logs/bleu_warmlin.txt",
}

os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def read_jsonl(path):
    out=[]
    if not os.path.exists(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out

def parse_bleu(logfile):
    if not os.path.exists(logfile): return None
    txt=open(logfile, "r", encoding="utf-8").read()
    # sacrebleu stdout line example 
    m=re.search(r"BLEU[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE)
    return float(m.group(1)) if m else None

# Collect data
epochs = {}
steps = {}
bleu = {}
for name, path in RUNS:
    epochs[name] = read_jsonl(os.path.join(path, "log_epochs.jsonl"))
    steps[name]  = read_jsonl(os.path.join(path, "log_steps.jsonl"))
    bleu[name]   = parse_bleu(BLEU_LOGS[name])

# --- Plot PPL by epoch ---
plt.figure()
for name in ["nowarm","warmconst","warmlin"]:
    y=[e["ppl_valid"] for e in epochs[name]] if epochs[name] else []
    x=list(range(len(y)))
    if y: plt.plot(x, y, label=name)
plt.xlabel("epoch"); plt.ylabel("valid perplexity")
plt.title("Perplexity by epoch")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ppl_by_epoch.png")
plt.close()

# --- Plot train loss by epoch ---
plt.figure()
for name in ["nowarm","warmconst","warmlin"]:
    y=[e["train_loss"] for e in epochs[name]] if epochs[name] else []
    x=list(range(len(y)))
    if y: plt.plot(x, y, label=name)
plt.xlabel("epoch"); plt.ylabel("train loss")
plt.title("Train loss by epoch")
plt.legend()
plt.tight_layout()
plt.savefig("plots/trainloss_by_epoch.png")
plt.close()

# --- Plot LR by step (first few thousand) ---
plt.figure()
for name in ["nowarm","warmconst","warmlin"]:
    s=steps[name][:3000] if steps[name] else []
    y=[r["lr"] for r in s]
    x=list(range(len(y)))
    if y: plt.plot(x, y, label=name)
plt.xlabel("step (truncated)")
plt.ylabel("learning rate")
plt.title("Learning rate by step")
plt.legend()
plt.tight_layout()
plt.savefig("plots/lr_by_step.png")
plt.close()

# --- Make a Markdown report ---
def last_epoch(d):
    return d[-1] if d else {}
def last_bleu(name):
    return bleu.get(name)

def nicely(x):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.3g}"
    return str(x)

rows=[]
for name in ["nowarm","warmconst","warmlin"]:
    le=last_epoch(epochs[name])
    rows.append({
        "run": name,
        "final_valid_ppl": le.get("ppl_valid"),
        "final_train_loss": le.get("train_loss"),
        "epoch_time_s": le.get("epoch_time_s"),
        "bleu_test": last_bleu(name)
    })

md = []
md.append("# Assignment 3 – Warmup Retraining \n")
md.append("## Final scores\n")
md.append("| Run | Valid PPL | Train Loss | Time/Epoch (s) | Test BLEU |")
md.append("|---|---:|---:|---:|---:|")
for r in rows:
    md.append(f"| {r['run']} | {nicely(r['final_valid_ppl'])} | {nicely(r['final_train_loss'])} | {nicely(r['epoch_time_s'])} | {nicely(r['bleu_test'])} |")

md.append("\n## Plots\n")
for fig in ["plots/ppl_by_epoch.png", "plots/trainloss_by_epoch.png", "plots/lr_by_step.png"]:
    if os.path.exists(fig):
        md.append(f"![{fig}]({os.path.basename(fig)})")

open("reports/assignment3_report.md","w",encoding="utf-8").write("\n".join(md))
print("Wrote plots/ and reports/assignment3_report.md")
