# Released simulation results

The two directories contain 10,000 fixed and 10,000 switching trials per policy. Each compressed JSON-lines record is a scalar projection of one completed native rollout. The original action checkpoint SHA-256 was verified against the archived evaluation configuration. The switching journals also retain the frozen schedule identifier, schedule digest and number of applied interventions. The public simulator writes those fields and the complete intervention events.

| Policy | Training export SHA-256 | Safetensors SHA-256 | EMA tensors |
| --- | --- | --- | ---: |
| Regression 10k | `737e5706fb3a97a3aca61c4166cd2320bb5facad0a5301b0860aba0c4f518a4d` | `4f493ad2cd29598809cbeb6c8d93fa63bc6f5df19e20866511a69f6990dd28b2` | 292 |
| Flow 20k | `ebcb2df9f06ef30d6785aa1dcf642d46ab36787eaef8f34c5690026ba2e520ff` | `0af867f4d5f20498f1bf74b064849acd1b3c653886245133da8fcfb583844ee6` | 298 |

The release export preserves all EMA tensor values, the model configuration, train-only statistics and task order. The scorer uses the paper's common action budget. The 800 ablation outcomes in `ablation-episodes.jsonl.gz` cover four policies, 40 tasks, one physical start per task and five lighting regimes. The full-model reference is the initial-index-zero subset of the main fixed-lighting evaluation. See the [simulation guide](../../../../docs/paper-reproduction.md) for commands and the checkpoint selection rule.
