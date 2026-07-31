---
search:
  exclude: true
---
# GARVIS ローカル ラティス認知サイクル

 **Adrien D. Thomas** の指揮の下、 **ProCityHub** として執筆されました。

ローカルのラティスサイクル・モードは、明示的に与えられた JSON のエビデンス・エンベロープを、次の決定的シーケンスで処理します:

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

このモードは OpenAI API キーを必要とせず、エビデンスを LLM に送信しません。

`--external-action` は外部提案を評価しますが、実行はしません。該当する提案は人によるレビューが必要です。

## 正準ハートビート正規化

```text
1.0 + 0.6 = 1.6
1.6 normalized to center = 1.0
```

出力には、決定的なエビデンス、パルス、リコール、平衡、完全サイクルのハッシュが含まれます。

## 境界

これは古典的な決定論的エンジニアリングモデルです。生物学的記憶、意識、感受性、AGI、量子的挙動、霊的メカニズム、臨床心理学、または普遍的真理の証明ではありません。ネットワーク、コネクタ、センシング、ツール、外部実行のいかなる権限も付与しません。