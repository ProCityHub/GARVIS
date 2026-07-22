---
search:
  exclude: true
---
# GARVIS ローカル ラティス 認知サイクル

 **Adrien D. Thomas** の指示の下で著述され、 **ProCityHub** として活動しています。

ローカルのラティスサイクル・モードは、明示的に提供された JSON 証拠エンベロープを次の決定論的シーケンスで処理します:

```text
evidence
→ psychology assessment
→ recurrent lattice-memory consolidation
→ Hypercube Heartbeat pulse
→ associative recall
→ equilibrium evaluation
→ bounded proposal status
```

## ローカル実行

```bash
env -u OPENAI_API_KEY \
  PYTHONPATH="$PWD/src:$PWD" \
  python -m garvis.cli \
  --lattice-cycle examples/lattice_cycle/evidence.example.json \
  --cycle 1 \
  --external-action
```

このモードは OpenAI API キーを必要とせず、証拠を LLM に送信しません。

`--external-action` は外部の提案を評価しますが、決して実行しません。適格な提案には人によるレビューが必要です。

## カノニカル ハートビート正規化

```text
1.0 + 0.6 = 1.6
1.6 normalized to center = 1.0
```

出力には、決定論的な証拠、パルス、リコール、平衡、および完全サイクルのハッシュが含まれます。

## 境界

これは古典的な決定論的エンジニアリングモデルです。生物学的記憶、意識、感受性 (sentience)、 AGI 、量子挙動、スピリチュアルな機構、臨床心理学、または普遍的真理の証明ではありません。ネットワーク、コネクタ、センシング、ツール、または外部実行の権限を付与しません。