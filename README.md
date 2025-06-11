# Awesome Deep Search Papers

This repository is dedicated to collecting papers on **Deep Search**, an emerging area that focuses on the integration of **search** and **reasoning**. The goal is to gather and categorize works that leverage the synergy between information seeking and reasoning abilities to solve complex problems. The works collected here cover various concepts, including but not limited to **search agent**, **agentic RAG**, **Deep Research**, and **search-enhanced reasoning models**.

The papers are primarily divided into two main categories:
- **RL-based Methods**: Approaches that utilize Reinforcement Learning to train agents for web-search and reasoning tasks.
- **Non-RL-based Methods**: Approaches that achieve deep search capabilities without relying on Reinforcement Learning, including training-free or Supervised Fine-Tuning (SFT) solutions.

If you find this list helpful, contributions are welcome via pull requests.

---

## Table of Contents

- [🧠 Methods](#methods)
  - [🔁 RL-based Methods](#rl-based-methods)
  - [📘 Non-RL-based Methods](#non-rl-based-methods)
- [📊 Benchmarks & Datasets](#benchmarks--datasets)
  - [Multi-Hop QA Dataset](#multi-hop-qa-dataset)
  - [Challenging QA for Deep Search](#challenging-qa-for-deep-search)
  - [Open-domain QA for Deep Research](#open-domain-qa-for-deep-research)

---

## Methods

### RL-based Methods
| Time   | Title                                                                                                                              | Venue     |
| :----- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------- |
| 2025.6 | [Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification](https://www.arxiv.org/abs/2506.07528)      | arXiv     |
| 2025.6 | [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2506.04185)           | arXiv     |
| 2025.5 | [Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/abs/2505.11277)                 | arXiv     |
| 2025.5 | [Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging](https://arxiv.org/abs/2505.09316)                 | arXiv     |
| 2025.5 | [An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/abs/2505.15117)          | arXiv     |
| 2025.5 | [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.22019) | arXiv     |
| 2025.5 | [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501)                                            | arXiv     |
| 2025.5 | [ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.15776) | arXiv     |
| 2025.5 | [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](https://arxiv.org/abs/2505.14069)                | arXiv     |
| 2025.5 | [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005) | arXiv     |
| 2025.5 | [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/abs/2505.24332)            | arXiv     |
| 2025.5 | [MaskSearch: A Universal Pre-Training Framework to Enhance Agentic Search Capability](https://arxiv.org/abs/2505.20285)              | arXiv     |
| 2025.5 | [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107)                | arXiv     |
| 2025.5 | [Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](https://arxiv.org/abs/2505.12065)              | arXiv     |
| 2025.5 | [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648)                                          | arXiv     |
| 2025.5 | [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588)                            | arXiv     |
| 2025.5 | [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582)          | arXiv     |
| 2025.3 | [Agent models: Internalizing Chain-of-Action Generation into Reasoning models](https://arxiv.org/abs/2503.06580)                      | arXiv     |
| 2025.4 | [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776)                       | arXiv     |
| 2025.4 | [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736)                                | arXiv     |
| 2025.4 | [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)     | arXiv     |
| 2025.3 | [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)                      | arXiv     |
| 2025.3 | [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592)                | arXiv     |
| 2025.3 | [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)        | arXiv     |

### Non-RL-based Methods
| Time    | Title                                                                                                                               | Venue      |
| :------ | :---------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| 2025.6  | [Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](https://arxiv.org/abs/2506.02454) | arXiv      |
| 2025.5  | [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834)       | arXiv      |
| 2025.5  | [AutoData: A Multi-Agent System for Open Web Data Collection](https://arxiv.org/abs/2505.15859)                                       | arXiv      |
| 2025.5  | [ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework](https://arxiv.org/abs/2505.18105) | arXiv      |
| 2025.3  | [Open deep search: Democratizing search with open-source reasoning agents.](https://arxiv.org/abs/2503.20201)                          | arXiv      |
| 2025.2  | [Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking](https://arxiv.org/abs/2502.04751)                        | arXiv      |
| 2025.2  | [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/abs/2502.04644)                                  | arXiv      |
| 2025.2  | [Atom of Thoughts for Markov LLM Test-Time Scaling](https://arxiv.org/abs/2502.12018)                                                  | arXiv      |
| 2025.1  | [Search-o1: Agentic search-enhanced large reasoning models](https://arxiv.org/abs/2501.05366)                                           | arXiv      |
| 2025.1  | [AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation using Tree-based Search](https://arxiv.org/abs/2501.10053)   | arXiv      |
| 2025.1  | [ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding](https://arxiv.org/abs/2501.07861)                            | arXiv      |
| 2025.1  | [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342)                                                             | arXiv      |
| 2024.12 | [Level-Navi Agent: A Framework and benchmark for Chinese Web Search Agents](https://arxiv.org/abs/2502.15690)                             | arXiv      |
| 2024.11 | [Auto-rag: Autonomous retrieval-augmented generation for large language models.](https://arxiv.org/abs/2411.19443)                      | arXiv      |
| 2024.10 | [Plan*rag: Efficient test-time planning for retrieval augmented generation.](https://arxiv.org/abs/2410.20753)                           | arXiv      |
| 2024.10 | [Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](https://arxiv.org/abs/2410.01428) | arXiv      |
| 2024.10 | [Inference scaling for long-context retrieval augmented generation.](https://arxiv.org/abs/2410.04343)                                   | ICLR 2025  |
| 2024.7  | [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183)                                           | arXiv      |
| 2023.12 | [KwaiAgents: Generalized Information-seeking Agent System with Large Language Models](https://arxiv.org/abs/2312.04889)                   | WWW 2024   |
| 2023.12 | [ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003)                                 | ICLR 2024  |
| 2023.10 | [Self-rag: Learning to retrieve, generate, and critique through self-reflection](https://arxiv.org/abs/2310.11511)                        | ICLR 2024  |
| 2023.5  | [Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources](https://arxiv.org/abs/2305.13269) | ICLR 2024  |
| 2022.12 | [Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.](https://aclanthology.org/2023.acl-long.557.pdf) | ACL 2023   |

---

## Benchmarks & Datasets

### Multi-Hop QA Dataset

| Name            | Venue       | Link |
|------------------|-------------|------|
| HotpotQA         | EMNLP 2018  | [http://arxiv.org/abs/1809.09600](http://arxiv.org/abs/1809.09600) |
| 2WikiMultiHopQA  | COLING 2020 | [https://arxiv.org/abs/2011.01060](https://arxiv.org/abs/2011.01060) |
| Bamboogle        | EMNLP 2023  | [https://arxiv.org/abs/2210.03350](https://arxiv.org/abs/2210.03350) |
| MuSiQue          | TACL 2022   | [https://arxiv.org/abs/2108.00573](https://arxiv.org/abs/2108.00573) |
| StrategyQA       | TACL 2021   | [https://arxiv.org/abs/2101.02235](https://arxiv.org/abs/2101.02235) |
| FRAMES           | NAACL 2025  | [https://arxiv.org/abs/2409.12941](https://arxiv.org/abs/2409.12941) |
| MultiHop-RAG     | COLM 2024   | [https://arxiv.org/abs/2401.15391](https://arxiv.org/abs/2401.15391) |

---

### Challenging QA for Deep Search

| Name           | Venue      | Link |
|----------------|------------|------|
| BrowseComp     | arXiv 2025 | [https://arxiv.org/abs/2504.12516](https://arxiv.org/abs/2504.12516) |
| InfoDeepSeek   | arXiv 2025 | [https://arxiv.org/abs/2505.15872](https://arxiv.org/abs/2505.15872) |
| ORION          | arXiv 2025 | [https://arxiv.org/abs/2505.18105](https://arxiv.org/abs/2505.18105) |
| BrowseComp-ZH  | arXiv 2025 | [https://arxiv.org/abs/2504.19314](https://arxiv.org/abs/2504.19314) |
| Web24          | arXiv 2024 | [https://arxiv.org/abs/2502.15690](https://arxiv.org/abs/2502.15690) |

---

### Open-domain QA for Deep Research

| Name                | Venue      | Link |
|---------------------|------------|------|
| O2-QA               | arXiv 2025 | [https://arxiv.org/abs/2505.16582](https://arxiv.org/abs/2505.16582) |
| Researchy Questions | arXiv 2024 | [https://arxiv.org/abs/2402.17896](https://arxiv.org/abs/2402.17896) |
| MultimodalReportBench | arXiv 2025 | [https://arxiv.org/abs/2506.02454](https://arxiv.org/abs/2506.02454) |
| DeepResearchGym     | arXiv 2025 | [https://arxiv.org/abs/2505.19253](https://arxiv.org/abs/2505.19253) |
| Deep Research Bench | arXiv 2025 | [https://www.arxiv.org/abs/2506.06287](https://www.arxiv.org/abs/2506.06287) |

---

Feel free to open an issue or PR to add new papers and benchmarks!
