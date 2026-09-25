# StudyVision Evaluation Results

## Dataset

Total sessions: 18
Total samples: 48,111

Activities:
- mixed_study
- paper_study
- screen_study
- typing_study

## Leave-One-Session-Out Evaluation

### Frame-Level Evaluation

Mean session accuracy: 0.4977
Sample-weighted accuracy: 0.4826

### Window-Level Evaluation

Window size: 30 samples

Mean session accuracy: 0.5451
Window-weighted accuracy: 0.5223

## Activity-Level Window Results

| Activity | Sessions | Mean Accuracy |
|---|---:|---:|
| mixed_study | 4 | 0.3548 |
| paper_study | 6 | 0.8008 |
| screen_study | 4 | 0.4892 |
| typing_study | 4 | 0.4080 |

## Held-Out Validation Sessions

| Session | Activity | Windows | Accuracy |
|---|---|---:|---:|
| validation_001 | paper_study | 28 | 0.8214 |
| validation_002 | paper_study | 28 | 0.7143 |

## Best Individual Session

paper_003
Accuracy: 0.9444 (94.44%)

## Interpretation

The 30-sample window evaluation is more stable than frame-level prediction,
but its overall score also shows that the current eye-only feature set does not
reliably separate every study activity across unseen sessions. StudyVision is
therefore reported as an observed-activity prototype, rather than a direct
measure of attention.
