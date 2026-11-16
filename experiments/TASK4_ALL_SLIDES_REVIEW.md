# Task 4: review of all seven lecture decks

Follow-up: Item2vec was subsequently implemented for Task4 in task4_item2vec.py; the unimplemented-model table below records the earlier review state. See TASK4_ITEM2VEC.md for the new experiment.

All seven local PDF blob hashes match the latest `main/Slides` files on GitHub. The folder contains classification (118 pages), content and structure (117), ethics (95), recommendation (207), regression (107), temporal (123), and text mining (111). The review indexed every slide and checked the relevant model sections. Slides 1–3 of content and structure, titled Assignment 1 tips, are blank below the heading in the supplied PDF.

Task 4 predicts which of two supplied books corresponds to a reader's interaction at a supplied timestamp. Training provides reader, book, timestamp and rating. User/book IDs are opaque. We can derive histories, co-occurrence graphs and chronological context from provided training rows; we do not have book descriptions, review text, covers, locations, demographics or friendships. No external Goodreads collection or ID decoding is allowed. Observed zero-rating interactions remain positive reading events. If ratings are used for an auxiliary model, zero can be treated as a missing rating while retaining the reading event.

## Coverage of every deck

| Deck | Relevant material | Current status / limits |
|---|---|---|
| classification.pdf | Naive Bayes pp20–39; logistic regression pp40–53; SVM pp54–66 | Logistic regression already used in Task4 calibration/baselines. SVM and Naive Bayes are feasible new combiners on engineered pair features; SVM is more plausible than Naive Bayes with correlated scores. |
| content and structure.pdf | FM pp9–27, including FISM pp20–22; socially-aware models pp40–50; matching/advertising later | Temporal FM implemented. FISM remains unimplemented. Original social/content-dependent and global allocation methods require missing inputs or solve another objective. |
| ethics.pdf | Item2vec example p34; MMR pp22–25, DPP pp26–27, fairness/calibration | Item2vec is compatible via interactions. Diversity/list reranking and fairness objectives do not directly solve this fixed-pair accuracy task; many examples need attributes not supplied. |
| recommendation.pdf | Neighborhoods, matrix factorization, implicit feedback/BPR; SLIM pp165–168; FISM pp169–172; NCF pp180–182; AutoRec pp183–185; neural references p188; implicit-history extension pp194–195 | Many base methods implemented. FISM, NCF, AutoRec and explicit history-augmented learned factorization remain possible. Ordinary rating models need adaptation or use as auxiliary scores. |
| regression.pdf | Polynomial transforms pp42–44; temporal/periodic/piecewise features pp58–73; regularization pp95–100 | Much of this is already used via time features and regularization. Additional spline/piecewise and periodic interactions are compatible feature extensions, not a new recommender family. |
| temporal.pdf | Koren temporal factorization pp30–41; FPMC pp50–61; PRME pp64–77; playlist metric models pp79–91; user/community expertise evolution pp93–104; translation pp106–116; RNN pp118–123 | Drift partly implemented, but full joint temporal factorization and the listed sequence/metric/experience models are absent from Task4. Some sequence/metric models were implemented for Task5 only. |
| text mining.pdf | TF-IDF pp37–57; LSA pp58–68; Word2vec/Item2vec pp69–80 | Item2vec is directly adaptable to book-ID sequences. TF-IDF weighting plus SVD can be adapted to the interaction matrix. Text-only methods require unavailable review text. RNN/Transformer families are mentioned on p80, not taught in implementation detail. |

## Compatible models not yet implemented for Task 4

| Model | Exact slide source | Adaptation and caveats | Priority |
|---|---|---|---|
| Item2vec / skip-gram book embeddings | text mining pp73–78; ethics p34 | Train on chronological book-ID windows, score candidates against history near the query date. No review text is needed. | High |
| FISM | recommendation pp169–172; content and structure pp20–22 | Train separate history/target book embeddings; aggregate the user's history rather than a free user vector. A time-weighted pooling variant is our adaptation. The current FM does not contain pooled history embeddings, so it is not already FISM. | High |
| FPMC | temporal pp50–61 | Add a transition score from the immediately preceding known book to the candidate, plus personal preference. For Task4 locate the predecessor relative to the query timestamp. | High |
| Translation-based recommendation / TransRec-style | temporal pp106–116 | Represent a reader as a transition/translation vector between previous and candidate book embeddings. Task5 implementation does not imply Task4 implementation. | Medium |
| PRME-style / playlist metric embeddings | temporal pp64–91 | Use reader preference and adjacent-book distances. Original geographic POI components cannot be used without location data; adapt only the non-geographic sequential components. | Medium |
| Neural Collaborative Filtering | recommendation pp180–182 | Train raw reader/book embeddings with a nonlinear interaction function and implicit-feedback loss; optionally condition on time. Our neural score combiners are not this model. | Medium |
| AutoRec / implicit autoencoder adaptation | recommendation pp183–185 | Learn to reconstruct reading vectors. Use an implicit-feedback loss/masking scheme; time-local inputs are an additional adaptation. | Medium |
| Full jointly trained temporal factorization, plus implicit-history augmentation | temporal pp30–41; recommendation pp194–195 | Jointly learn changing reader factors, candidate-dependent temporal biases and history embeddings with an interaction/ranking loss. Current frozen-base drift is a partial implementation, not full TimeSVD++. | Medium |
| Learned expertise / experience-stage factorization | temporal pp93–104 | Let latent preference stages progress through each reader's history. The lecture version is rating-based; an interaction/ranking adaptation needs substantial new fitting code. | Exploratory |
| SVM on pair features | classification pp54–66 | Learn a maximum-margin classifier on differences/sums of the existing component scores; a pairwise-ranking variant is an adaptation. | Lower: changes the combiner rather than adding a new recommender signal |
| Naive Bayes on engineered/binned features | classification pp20–39 | Feasible classifier but conditional independence is poorly matched to our highly correlated component scores. | Low |
| RNN / GRU / LSTM; self-attention family | temporal pp118–123; recommendation p186; text mining p80 | Build a chronology-specific context for each query; optimize interaction/ranking loss. RNNs are discussed; specific GRU4Rec/SASRec architectures are adaptations beyond the detailed lecture material. | Later: higher training/implementation cost |
| Neural FM / DeepFM / Wide & Deep | recommendation p188 | Mentioned as further reading, not fully derived in the deck. Could add nonlinear interactions and time to the current FM inputs. | Exploratory |
| Rating-aware Pearson / rating-factorization auxiliary scores | recommendation pp32–37 and factorization chapters | Training ratings can supply an additional signal, but the objective remains interaction accuracy, not rating MSE. Keep zero-rated reading events while avoiding interpreting zero as an explicit dislike. | Lower |
| TF-IDF weighted interaction embeddings / truncated SVD adaptation | text mining pp37–68 | Readers act like documents and books like tokens; low-rank co-reading features require no actual text. This is a transfer of the text method to interactions. | Lower |
| Autoregression of binned book activity | temporal pp4–15 | Predict temporal book activity from training-only count series as another feature. Not a direct per-reader classifier. | Lower; existing KDE/window features overlap |

## Important mathematical compatibility detail

A pure reader-only additive bias b_u(t) cancels when comparing the two candidates at the same timestamp:

s(u,i1,t) - s(u,i2,t).

Therefore, adding only per-reader daily biases cannot improve Task4 choices. Temporal terms must affect books differently: reader-time embeddings interacting with book factors, book-time biases, candidate-dependent scaling, or sequential history/candidate interactions. A rating-focused temporal model should not be copied unchanged and assumed to optimize this task.

Pearson and rating-aware features are possible from training ratings, but user/item descriptions, geographic signals, friendship networks, images, genres and review text are unavailable. MMR/DPP and ad matching optimize diversity/resource allocation rather than identifying the positive book in each supplied pair.

## Recommended next sequence

1. Item2vec and time-weighted FISM: new learned book-context/history signals.
2. Query-time FPMC: personalized immediate-transition signal.
3. NCF or an implicit AutoRec model if the first two families do not provide enough complementarity.

Add their scores to the existing feature bank and compare fresh combiners/ensembles with the accepted public0.9485 LightGBM solution. Keep local tuning separate from a blend/model check. No prospective gain is guaranteed.
