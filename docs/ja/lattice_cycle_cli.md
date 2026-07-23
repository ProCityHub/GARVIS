---
search:
  exclude: true
---
# GARVIS ローカル ラティス 認知サイクル

**Adrien D. Thomas** の指揮のもと、 **ProCityHub** として執筆されました。

ローカルのラティスサイクル・モードは、明示的に与えられた JSON 証拠エンベロープを、次の決定的なシーケンスで処理します:

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

このモードは OpenAI の API key を必要とせず、証拠を LLM に送信しません。

`--external-action` は外部提案を評価しますが、決して実行しません。対象となる提案には人によるレビューが必要です。

## 正準ハートビート正規化

```text
1.0 + 0.6 = 1.6
1.6 normalized to center = 1.0
```

出力には、決定的な証拠、パルス、リコール、エクイリブリアム、そして完全サイクルのハッシュが含まれます。

## 境界

これは古典的な決定論的エンジニアリングモデルです。生物学的記憶、意識、感性、AGI、量子的挙動、霊的メカニズム、臨床心理学、または普遍的真理の証明ではありません。ネットワーク、コネクタ、センシング、ツール、または外部実行の権限も付与しません。