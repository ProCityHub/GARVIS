---
search:
  exclude: true
---
# GARVIS ローカル ラティス 認知サイクル

 **Adrien D. Thomas** の指揮の下、 **ProCityHub** として執筆されました。

ローカル ラティスサイクル モードは、明示的に提供された JSON 証拠エンベロープ（evidence envelope）を、この決定論的シーケンスで処理します:

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

このモードは OpenAI の API キーを必要とせず、証拠を LLM に送信しません。

`--external-action` は外部の提案を評価しますが、実行はしません。適格な提案には人によるレビューが必要です。

## 正準ハートビート正規化

```text
1.0 + 0.6 = 1.6
1.6 normalized to center = 1.0
```

出力には、決定論的な証拠、パルス (pulse)、リコール (recall)、均衡 (equilibrium)、および完全サイクル ハッシュが含まれます。

## 境界

これは古典的な決定論的エンジニアリング モデルです。これは、生物学的記憶、意識、知覚能力、 AGI 、量子挙動、スピリチュアルなメカニズム、臨床心理学、あるいは普遍的真理の証明ではありません。ネットワーク、コネクタ、センシング、ツール、または外部実行の権限は一切与えません。