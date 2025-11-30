import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError  # ROOT warning 억제

# ─── 0) 입력 파일 & 출력 디렉토리 ──────────────────────────────────────
# (파일 이름, 레이블)
input_samples = [
    ("hist_ST.root",   "ST"),
    ("hist_ST_t.root", "ST_t"),
    ("hist_ST_tW.root","ST_tW"),
    ("hist_TT.root",   "TT"),
]

out_dir = "compare_ST_like_overlay"
os.makedirs(out_dir, exist_ok=True)


# ─── 1) 히스토그램 이름 리스트 얻기 (첫 파일 기준) ─────────────────────
def get_hist_names_from_root(root_file):
    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open file: {root_file}")
    plots_dir = f.Get("plots")
    if not plots_dir:
        raise RuntimeError(f"'plots' directory not found in {root_file}")

    names = []
    for key in plots_dir.GetListOfKeys():
        obj = key.ReadObj()
        # 1D 히스토그램만 (TH1 계열 & dimension=1)
        if obj.InheritsFrom("TH1") and obj.GetDimension() == 1:
            names.append(obj.GetName())
    f.Close()
    return names


hist_names = get_hist_names_from_root(input_samples[0][0])
print(f"Found {len(hist_names)} 1D histograms in 'plots/'")


# ─── 2) 각 히스토그램 이름에 대해 4개 샘플 오버레이 플롯 ───────────────
for hname in hist_names:
    print(f"▶ {hname}")

    hist_list   = []
    labels_list = []

    # 2-1) 각 샘플에서 같은 이름의 히스토그램 가져오기
    edges_ref = None
    skip_this_hist = False

    for fname, label in input_samples:
        f = ROOT.TFile.Open(fname)
        if not f or f.IsZombie():
            print(f"   ✖ Cannot open {fname}, skip this file for {hname}")
            continue

        h = f.Get(f"plots/{hname}")
        if not h:
            # 이 파일에는 이 히스토그램이 없음
            f.Close()
            continue

        # 독립적으로 쓰기 위해 clone
        h_clone = h.Clone(f"{hname}_{label}_clone")
        h_clone.SetDirectory(0)
        f.Close()

        # bin 정보 추출
        nb = h_clone.GetNbinsX()
        edges = np.array(
            [h_clone.GetBinLowEdge(i) for i in range(1, nb + 1)]
            + [h_clone.GetBinLowEdge(nb) + h_clone.GetBinWidth(nb)]
        )
        counts = np.array([h_clone.GetBinContent(i) for i in range(1, nb + 1)])

        # 첫 샘플의 binning을 기준으로, 나머지가 다 같아야 함
        if edges_ref is None:
            edges_ref = edges
        else:
            if not np.allclose(edges_ref, edges):
                print(f"   ✖ Binning mismatch in {fname} for {hname}, skip this histogram")
                skip_this_hist = True
                break

        hist_list.append(counts)
        labels_list.append(label)

    if skip_this_hist:
        continue

    # 최소 2개 이상 있어야 비교 의미 있음
    if len(hist_list) < 2:
        print(f"   ✖ Less than 2 samples for {hname}, skipping")
        continue

    # ─── 2-2) NumPy로 정리 ────────────────────────────────────────────
    hist_arr   = np.array(hist_list)      # shape: (n_samples, nbins)
    n_samples  = hist_arr.shape[0]
    nbins      = hist_arr.shape[1]
    edges      = edges_ref
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # ─── 2-3) 메인 패널 + ratio 패널 레이아웃 ─────────────────────────
    plt.style.use(hep.style.CMS)
    fig, (ax, axr) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        figsize=(10, 8),
        dpi=150,
    )

    # ─── 2-4) 메인 패널: 샘플별 step 히스토그램 오버레이 ─────────────
    for counts, label in zip(hist_arr, labels_list):
        hep.histplot(
            counts,
            bins=edges,
            histtype="step",  # 스택 아님, 라인만
            label=label,
            ax=ax,
        )

    ax.set_ylabel("Events / bin")
    hep.cms.label("Private Work", data=False, year="2018", ax=ax)
    ax.legend(
        loc="upper right",
        prop={"size": 8},
        handletextpad=0.2,
        labelspacing=0.3,
        columnspacing=0.5,
    )

    # ─── 2-5) Ratio 패널: 첫 샘플(ST)을 기준으로 나머지 / ST ──────────
    ref = hist_arr[0]  # 기준: 첫 번째 샘플 (ST)
    ref_label = labels_list[0]

    # 기준 샘플은 1로 그려주고, 나머지 샘플은 ratio
    mc_mask = ref > 0

    for i in range(n_samples):
        counts = hist_arr[i]
        label  = labels_list[i]

        if i == 0:
            # 기준은 항상 1이므로 따로 그리지 않아도 됨 (축만 1에 맞춰줌)
            continue

        ratio = np.divide(
            counts,
            ref,
            out=np.ones_like(counts),
            where=mc_mask,
        )
        # step 스타일로 ratio 그리기
        hep.histplot(
            ratio,
            bins=edges,
            histtype="step",
            label=f"{label}/{ref_label}",
            ax=axr,
        )

    axr.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axr.set_ylabel("Ratio")
    axr.set_xlabel(hname)
    axr.set_ylim(0.5, 1.5)
    axr.legend(
        loc="upper right",
        prop={"size": 7},
        handletextpad=0.2,
        labelspacing=0.3,
        columnspacing=0.5,
    )

    # ─── 2-6) 저장 ──────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"compare_{hname}.png")
    fig.savefig(out_path)
    plt.close(fig)

print("✅ All overlay plots saved in", out_dir)

