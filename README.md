
# Awesome Search Agent Papers

This repository aims to collect and organize the latest research papers on **Search Agents**. Search agents leverage dynamic planning and in-depth information retrieval capabilities of intelligent agents, enabling them to dynamically adjust their search plans based on context for more efficient and accurate information acquisition.

This repository covers areas including, but not limited to **Search Agent**, **Agentic RAG (Retrieval-Augmented Generation)**, **Deep Research**, **Deep Search**, **Search-enhanced Reasoning Models**.

We broadly categorize current solutions into the following types:
* **Early Iterative Retrieval**: Papers exploring early mechanisms of iterative retrieval.
* **Tuning-free Methods**: Approaches that achieve search agent functionality without extensive specific training data.
* **SFT-based Methods**: Methods that train search agents using Supervised Fine-Tuning.
* **RL-based Methods**: Methods that train search agents using Reinforcement Learning.

For a deeper look, check out our survey paper: [A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges](https://arxiv.org/abs/2508.05668). If you find this repository helpful, please cite our survey paper.

```
@article{xi2025survey,
  title={A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges},
  author={Xi, Yunjia and Lin, Jianghao and Xiao, Yongzhao and Zhou, Zheli and Shan, Rong and Gao, Te and Zhu, Jiachen and Liu, Weiwen and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2508.05668},
  year={2025}
}
```

## Table of Contents

* [Methods](#methods)
    * [Early Iterative Retrieval](#early-iterative-retrieval)
    * [Tuning-free Methods](#tuning-free-methods)
    * [SFT-based Methods](#sft-based-methods)
    * [RL-based Methods](#rl-based-methods)
* [Datasets](#datasets)
    * [Multi-Hop QA Dataset](#multi-hop-qa-dataset)
    * [Challenging QA for Deep Search](#challenging-qa-for-deep-search)
    * [Fact-checking dataset](#fact-checking-dataset)
    * [Open-domain QA for Deep Research](#open-domain-qa-for-deep-research)
    * [Domain-specific dataset](#domain-specific-dataset)
    * [Other Aspect](#other-aspect)

---

## Methods

### Early Iterative Retrieval

| Time    | Paper Title                                                                                                                                                                      | Venue       |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
| 2025.6  | [Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation](https://arxiv.org/abs/2506.11092)                                   | arXiv       |
| 2025.4  | [Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding](https://arxiv.org/abs/2504.01281)                          | arXiv       |
| 2025.4  | [Credible plan-driven RAG method for Multi-hop Question Answering](https://arxiv.org/abs/2504.16787)                                                                             | arXiv       |
| 2025.3  | [Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning](https://arxiv.org/abs/2503.01642)                                                 | arXiv       |
| 2024.11 | [DMQR-RAG: Diverse Multi-Query Rewriting for RAG](https://arxiv.org/abs/2411.13154)                                                                                              | arXiv       |
| 2024.7  | [Adaptive Retrieval-Augmented Generation for Conversational Systems](https://arxiv.org/abs/2407.21712)                                                                          | NAACL 2025  |
| 2024.7  | [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach](https://arxiv.org/abs/2407.16833)                                               | EMNLP 2024  |
| 2024.7  | [REAPER: Reasoning based Retrieval Planning for Complex RAG Systems](https://arxiv.org/abs/2407.18553)                                                                          | CIKM 2024   |
| 2024.6  | [Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering](https://arxiv.org/abs/2406.14891)                                                       | ACL 2024    |
| 2024.6  | [Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs](https://arxiv.org/abs/2406.14282)                                                         | EMNLP 2024  |
| 2024.6  | [A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieval](https://arxiv.org/abs/2406.18960)                                        | SIGIR 2024  |
| 2024.3  | [RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation](https://arxiv.org/abs/2403.05313)                                                  | NeurIPS 2024|
| 2024.3  | [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)                                        | NAACL 2024  |
| 2024.3  | [Generating Multi-Aspect Queries for Conversational Search](https://arxiv.org/abs/2403.19302)                                                                                   | arXiv       |
| 2024.1  | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)                                                                                                    | arXiv       |
| 2023.5  | [Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy.](https://arxiv.org/abs/2305.15294)                                               | EMNLP 2023  |
| 2023.5  | [Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources](https://arxiv.org/abs/2305.13269)                                 | ICLR 2024   |
| 2023.5  | [Active retrieval augmented generation.](https://arxiv.org/abs/2305.06983)                                                                                                       | EMNLP 2023  |
| 2022.12 | [Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.](https://aclanthology.org/2023.acl-long.557.pdf)                              | ACL 2023    |
| 2022.12 | [DEMONSTRATE–SEARCH–PREDICT: Composing retrieval and language models for knowledge-intensive NLP](https://arxiv.org/abs/2212.14024)                                               | arXiv       |
| 2022.1  | [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)                                                                          | EMNLP 2023  |

### Tuning-free Methods

| Time    | Paper Title                                                                                                                                                                      | Venue         |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| 2025.9  | [Deep Research is the New Analytics System: Towards Building the Runtime for AI-Driven Analytics](https://arxiv.org/abs/2509.02751)  |  arXiv |
| 2025.9  | [L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search](https://arxiv.org/abs/2509.00761)  | arXiv |
| 2025.9  | [Universal Deep Research: Bring Your Own Model and Strategy](https://arxiv.org/abs/2509.00244)  | arXiv |
| 2025.8  | [You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures](https://arxiv.org/abs/2508.06105)  |  arXiv |
| 2025.8  | [Improving and Evaluating Open Deep Research Agents](https://arxiv.org/abs/2508.10152) | arXiv |
| 2025.8  | [BrowseMaster: Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair](https://arxiv.org/abs/2508.09129)  |  arXiv  |
| 2025.8  | [Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2508.08816) | arXiv |
| 2025.7  | [SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search](https://arxiv.org/abs/2507.15245)                                                              | arXiv         |
| 2025.7  | [Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications](https://arxiv.org/abs/2507.16507)                                                 | arXiv         |
| 2025.7  | [Deep Researcher with Test-Time Diffusion](https://arxiv.org/abs/2507.16075v1)                                                                                                  | arXiv         |
| 2025.7  | [Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search](https://arxiv.org/abs/2507.02652)                                                          | arXiv         |
| 2025.6  | [Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval](https://arxiv.org/abs/2506.17878)                                                          | arXiv         |
| 2025.6  | [Towards AI Search Paradigm](https://arxiv.org/abs/2506.17188)                                                                                                                   | arXiv         |
| 2025.6  | [KnowCoder-V2: Deep Knowledge Analysis](https://arxiv.org/abs/2506.06881)                                                                                                        | arXiv         |
| 2025.6  | [Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](https://arxiv.org/abs/2506.02454)                                     | arXiv         |
| 2025.6  | [From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents](https://arxiv.org/abs/2506.18959)                                                   | arXiv         |
| 2025.5  | [AutoData: A Multi-Agent System for Open Web Data Collection](https://arxiv.org/abs/2505.15859)                                                                                   | arXiv         |
| 2025.5  | [ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework](https://arxiv.org/abs/2505.18105)                             | arXiv         |
| 2025.5  | [Code Researcher: Deep Research Agent for Large Systems Code and Commit History](https://arxiv.org/abs/2506.11060)                                                              | arXiv         |
| 2025.5  | [MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.20096)                                               | arXiv         |
| 2025.5  | [ITERKEY: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation](https://arxiv.org/abs/2505.08450)                                                  | arXiv         |
| 2025.3  | [Open deep search: Democratizing search with open-source reasoning agents.](https://arxiv.org/abs/2503.20201)                                                                    | arXiv         |
| 2025.3  | [MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2503.20757)                                                              | arXiv         |
| 2025.3  | [Agentic RAG with Human-in-the-Retrieval](https://www.computer.org/csdl/proceedings-article/icsa-c/2025/333600a498/278QVgyUK2I)                                                   | ICSA-C 2025   |
| 2025.2  | [WebWalker: Benchmarking LLMs in Web Traversal](https://arxiv.org/abs/2501.07572)  | ACL 2025 |
| 2025.2  | [Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking](https://arxiv.org/abs/2502.04751)                                                                | arXiv         |
| 2025.2  | [ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents](https://arxiv.org/pdf/2502.18017)                                               | arxiv         |
| 2025.2  | [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/abs/2502.04644)                                                                           | arXiv         |
| 2025.2  | [An Agent Framework for Real-Time Financial Information Searching with Large Language Models](https://arxiv.org/abs/2502.15684)                                                  | arXiv         |
| 2025.2  | [DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](https://arxiv.org/abs/2502.20730)                                  | arXiv         |
| 2025.2  | [MCTS-KBQA: Monte Carlo Tree Search for Knowledge Base Question Answering](https://arxiv.org/abs/2502.13428)                                                                     | arXiv         |
| 2025.1  | [Search-o1: Agentic search-enhanced large reasoning models](https://arxiv.org/abs/2501.05366)                                                                                    | arXiv         |
| 2025.1  | [AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation using Tree-based Search](https://arxiv.org/abs/2501.10053)                                             | arXiv         |
| 2025.1  | [ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding](https://arxiv.org/abs/2501.07861)                                                                     | arXiv         |
| 2025.1  | [Retrieval-Augmented Generation by Evidence Retroactivity in LLMs](https://arxiv.org/abs/2501.05475)                                                                              | arXiv         |
| 2024.12 | [Level-Navi Agent: A Framework and benchmark for Chinese Web Search Agents](https://arxiv.org/abs/2502.15690)                                                                     | arXiv         |
| 2024.12 | [RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement](https://arxiv.org/abs/2412.12881)                                               | arXiv         |
| 2024.12 | [Progressive Multimodal Reasoning via Active Retrieval](https://arxiv.org/abs/2412.14835)                                                                                        | arXiv         |
| 2024.11 | [SRSA: A Cost-Efficient Strategy-Router Search Agent for Real-world Human-Machine Interactions](https://arxiv.org/abs/2411.14574)                                                | arXiv         |
| 2024.10 | [Plan*rag: Efficient test-time planning for retrieval augmented generation.](https://arxiv.org/abs/2410.20753)                                                                     | arXiv         |
| 2024.10 | [Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](https://arxiv.org/abs/2410.01428)                    | arXiv         |
| 2024.10 | [Inference scaling for long-context retrieval augmented generation.](https://arxiv.org/abs/2410.04343)                                                                            | ICLR 2025     |
| 2024.9  | [Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation](https://openreview.net/forum?id=g2C947jjjQ)                                                              | arxiv         |
| 2024.8  | [Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations](https://arxiv.org/abs/2408.15232)                                | EMNLP 2024    |
| 2024.8  | [Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering](https://arxiv.org/abs/2408.11875)                                              | arxiv         |
| 2024.7  | [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183)                                                                                   | arXiv         |
| 2024.7  | [Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach](http://arxiv.org/abs/2407.13101)                                                 | WWW2025 Agent4IR|
| 2024.6  | [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/abs/2406.12430)                                  | NAACL 2024    |
| 2024.4  | [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)                                                          | NAACL 2024    |
| 2024.2  | [Metacognitive Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2402.11626)                                                                                      | WWW 2024      |
| 2023.8  | [Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for Knowledge-intensive Question Answering](https://arxiv.org/abs/2308.13259)                                        | arXiv         |
| 2023.4  | [Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks](https://arxiv.org/abs/2304.14732)                                 | WWW 2024      |

### SFT-based Methods

| Time    | Paper Title                                                                                                                                                                      | Venue         |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| 2025.8   |  [Hybrid Deep Searcher: Integrating Parallel and Sequential Search Reasoning](https://arxiv.org/abs/2508.19113)  |  arXiv |
| 2025.8	 | [TURA: Tool-Augmented Unified Retrieval Agent for AI Search](https://arxiv.org/abs/2508.04604) | arXiv	|
| 2025.7	 | [Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training](https://arxiv.org/abs/2508.00414)	| arXiv	|
| 2025.5  | [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834)                                                  | arXiv         |
| 2025.5  | [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128)                                                            | arXiv         |
| 2025.4  | [KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](https://arxiv.org/abs/2501.18922)                                                              | ICML 2025     |
| 2025.3  | [ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation](https://arxiv.org/abs/2503.21729)              | arXiv         |
| 2025.2  | [RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](https://arxiv.org/pdf/2502.10996)                                                                        | arXiv         |
| 2025.1  | [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342)                                                                                                      | arXiv         |
| 2024.11 | [Auto-rag: Autonomous retrieval-augmented generation for large language models.](https://arxiv.org/abs/2411.19443)                                                                | arXiv         |
| 2024.10 | [Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models](https://arxiv.org/abs/2410.01782)                                                      | EMNLP 2025    |
| 2023.12 | [KwaiAgents: Generalized Information-seeking Agent System with Large Language Models](https://arxiv.org/abs/2312.04889)                                                          | WWW 2024      |
| 2023.12 | [ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003)                                                                        | ICLR 2024     |
| 2023.10 | [Self-rag: Learning to retrieve, generate, and critique through self-reflection](https://arxiv.org/abs/2310.11511)                                                                | ICLR 2024     |

### RL-based Methods

| Time    | Paper Title                                                                                                                                                                      | Venue         |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| 2025.9  | [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)  | arXiv |
| 2025.9  | [AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.08755)  |  arXiv |
| 2025.9  | [SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents](https://arxiv.org/abs/2509.06283) | arXiv |
| 2025.9  | [VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use](https://arxiv.org/abs/2509.01055)  |  arXiv |
| 2025.9  | [Open Data Synthesis For Deep Research](https://arxiv.org/abs/2509.00375)  | arXiv |
| 2025.8  | [Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](https://arxiv.org/abs/2508.20324)  | arXiv |
| 2025.8  | [AWorld: Orchestrating the Training Recipe for Agentic AI](https://arxiv.org/abs/2508.20404)  |  arXiv |
| 2025.8  | [AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2508.20368)  |  arXiv |
| 2025.8  | [Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153v2)  | arXiv  |
| 2025.8  | [OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](https://arxiv.org/abs/2508.16438)  | arXiv |
| 2025.8  | [Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](https://arxiv.org/abs/2508.13167)  | arXiv |
| 2025.8  | [MedReseacher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework](https://arxiv.org/abs/2508.14880)  | arXiv	|
| 2025.8  | [Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](https://arxiv.org/abs/2508.05748)  |  arXiv |
| 2025.8  | [WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent](https://arxiv.org/abs/2508.05748)  | arXiv |
| 2025.8  | [HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches](https://arxiv.org/abs/2508.08088) | arXiv |
| 2025.8  | [REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation](https://arxiv.org/abs/2508.08149)  | arXiv |
| 2025.8  | [Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL](https://arxiv.org/abs/2508.07976)  | arXiv | 
| 2025.8  | [SSRL: Self-Search Reinforcement Learning](https://arxiv.org/abs/2508.10874) | arXiv | 
| 2025.8  | [UR2: Unify RAG and Reasoning through Reinforcement Learning](https://arxiv.org/abs/2508.06165) | arXiv |
| 2025.8  | [ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning](https://arxiv.org/abs/2508.09303) | arXiv |	
| 2025.8  | [Lucy: edgerunning agentic web search on mobile with machine generated task vectors](https://arxiv.org/abs/2508.00360) | arXiv	|
| 2025.8  | [MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation](https://arxiv.org/abs/2508.01005)	| arXiv	|
| 2025.8  | [Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy](https://arxiv.org/abs/2508.01696) | arXiv |	
| 2025.8  | [GRAIL:Learning to Interact with Large Knowledge Graphs for Retrieval Augmented Reasoning](https://arxiv.org/abs/2508.05498)  |	arXiv |	
| 2025.7  | [Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849) | arXiv |
| 2025.7	 | [WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization](https://arxiv.org/abs/2507.15061)	| arXiv |	
| 2025.7  | [DynaSearcher: Dynamic Knowledge Graph Augmented Search Agent via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2507.17365)                                          | arXiv         |
| 2025.7  | [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592)                                                                                    | arXiv         |
| 2025.7  | [RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](https://arxiv.org/abs/2507.02962)                                            | arXiv         |
| 2025.6  | [Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification](https://www.arxiv.org/abs/2506.07528)                                                 | arXiv         |
| 2025.6  | [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2506.04185)                                                       | arXiv         |
| 2025.6  | [KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models](https://arxiv.org/abs/2506.19466)                                            | arXiv         |
| 2025.5  | [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246)  |  arXiv  |
| 2025.5  | [Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](https://arxiv.org/abs/2505.16410)  | arXiv |
| 2025.5  | [Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/abs/2505.11277)                                                              | arXiv         |
| 2025.5  | [Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty](https://arxiv.org/abs/2505.17281)                                                                | arXiv         |
| 2025.5  | [Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging](https://arxiv.org/abs/2505.09316)                                                           | arXiv         |
| 2025.5  | [An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/abs/2505.15117)                                                     | arXiv         |
| 2025.5  | [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.22019) | arXiv         |
| 2025.5  | [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501)                                                                                         | arXiv         |
| 2025.5  | [ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.15776)                             | arXiv         |
| 2025.5  | [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](https://arxiv.org/abs/2505.14069)                                                          | arXiv         |
| 2025.5  | [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005)                                            | arXiv         |
| 2025.5  | [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/abs/2505.24332)                                                       | arXiv         |
| 2025.5  | [MaskSearch: A Universal Pre-Training Framework to Enhance Agentic Search Capability](https://arxiv.org/abs/2505.20285)                                                          | arXiv         |
| 2025.5  | [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107)                                                          | arXiv         |
| 2025.5  | [Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](https://arxiv.org/abs/2505.12065)                                                       | arXiv         |
| 2025.5  | [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648)                                                                                      | arXiv         |
| 2025.5  | [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588)                                                                      | arXiv         |
| 2025.5  | [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582)                                                     | arXiv         |
| 2025.5  | [s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146)                                                                              | arXiv         |
| 2025.5  | [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596)                                             | arXiv         |
| 2025.4  | [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776)                                                                  | arXiv         |
| 2025.4  | [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736)                                                                          | arXiv         |
| 2025.4  | [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)                                                  | arXiv         |
| 2025.4  | [ReZero: Enhancing LLM Search Ability by Trying One More Time](https://arxiv.org/abs/2504.11001)                                                                                  | arXiv         |
| 2025.3  | [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)                                                                  | arXiv         |
| 2025.3  | [Agent models: Internalizing Chain-of-Action Generation into Reasoning models](https://arxiv.org/abs/2503.06580)                                                                | arXiv         |
| 2025.3  | [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592)                                                          | arXiv         |
| 2025.3  | [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)                                                    | arXiv         |
| 2025.2  | [DeepRetrieval: Hacking Real Search Engines and Retrievers with LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.00223)                                               | arXiv         |
| 2025.2  | [RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.13957)                                                       | arXiv         |
| 2025.2  | [DeepRAG: Thinking to Retrieval Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142)                                                                        | arXiv         |
| 2025.1  | [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2501.15228)                                                          | arXiv         |
| 2024.10 | [SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback](https://arxiv.org/pdf/2410.18141)                                                                      | ICLR 2025     |


## Datasets

### Multi-Hop QA Dataset

| Name          | Title                                                                                                                                                                      | Venue         |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| HotpotQA      | [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](http://arxiv.org/abs/1809.09600)                                                             | EMNLP 2018    |
| WikiMultiHopQA| [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://arxiv.org/abs/2011.01060)                                                  | COLING 2020   |
| Bamboogle     | [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)                                                                  | EMNLP 2023    |
| MuSiQue       | [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573)                                                                       | TACL 2022     |
| StrategyQA    | [Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies](https://arxiv.org/abs/2101.02235)                                        | TACL 2021     |
| FRAMES        | [Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2409.12941)                                                      | NAACL 2025    |
| MultiHop-RAG  | [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)                                                       | COLM 2024     |
| HoVer         | [HoVer: A dataset for many-hop fact extraction and claim verification](https://arxiv.org/abs/2011.03088)                                                                 | EMNLP 2020    |
| FanOutQA      | [FanOutQA: A Multi-Hop, Multi-Document Question Answering Benchmark for Large Language Models](https://arxiv.org/abs/2402.14116)                                           | ACL 2024      |
| Web24         | [Level-Navi Agent: A Framework and benchmark for Chinese Web Search Agents](https://arxiv.org/abs/2502.15690)                                                            | arXiv 2025    |
| ViDoRAG       | [ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents](https://arxiv.org/abs/2502.18017)                                        | arXiv 2025    |
| MoreHopQA     | [MoreHopQA: More Than Multi-hop Reasoning](https://arxiv.org/abs/2406.13397)                                                                                           | arXiv 2024    |
| CofCA         | [Cofca: A Step-Wise Counterfactual Multi-hop QA benchmark](https://arxiv.org/abs/2402.11924v5)                                                                           | ICLR 2025     |

### Challenging QA for Deep Search

| Name           | Title                                                                                                                                                                      | Venue         |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| BrowseComp     | [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://arxiv.org/abs/2504.12516)                                                                   | arXiv 2025    |
| InfoDeepSeek   | [InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation](https://arxiv.org/abs/2505.15872)                                             | arXiv 2025    |
| ORION          | [ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework](https://arxiv.org/abs/2505.18105)                     | arXiv 2025    |
| BrowseComp-ZH  | [BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese](https://arxiv.org/abs/2504.19314)                                                 | arXiv 2025    |
| PopQA          | [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/abs/2212.10511)                              | ACL 2023      |
| WebPuzzle      | [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/abs/2505.24332)                                               | arXiv 2025    |
| BLUR           | [Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](https://arxiv.org/abs/2503.19193)                                        | arXiv 2025    |
| BRIGHT         | [BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval](https://arxiv.org/abs/2407.12883)                                                      | ICLR 2025     |
| SealQA         | [SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](https://arxiv.org/abs/2506.01062)                                                           | arXiv 2025    |
| MMSearch        | [MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines](https://arxiv.org/abs/2409.12959)                                                   | arXiv 2024    |
| ScholarSearch  | [ScholarSearch: Benchmarking Scholar Searching Ability of LLMs](https://arxiv.org/abs/2506.13784)                                                                        | arXiv 2025    |
| Mind2Web 2     | [Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge](https://arxiv.org/abs/2506.21506)                                                                           | arXiv 2025    |
| BrowseComp-Plus | [BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent](https://arxiv.org/abs/2508.06600)  |  arXiv 2025 |
| MM-BrowseComp  | [MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents](https://arxiv.org/abs/2508.13186) | arXiv 2025 |
| WebShaperQA | [WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization](https://arxiv.org/abs/2507.15061) | arXiv 2025 |
| WebWalkerQA  | [WebWalker: Benchmarking LLMs in Web Traversal](https://arxiv.org/abs/2501.07572)  |  ACL 2025 |
| BrowseComp-VL  | [WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent](https://arxiv.org/abs/2508.05748) | arXiv 2025 |
| MAT-Search  | [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246) | arXiv 2025 |
| MMSearch-Plus  |  [MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents](https://arxiv.org/abs/2508.21475)  | arXiv 2025 |	


### Fact-checking dataset

| Name        | Title                                                                                                                                                                      | Venue           |
| :---------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- |
| LiveDRBench | [Characterizing Deep Research: A Benchmark and Formal Definition](https://arxiv.org/abs/2508.04183) | arXiv 2025	|
| Mocheg      | [End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models](https://arxiv.org/abs/2205.12487)                                      | SIGIR 2023      |
| MFC-Bench   | [MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models](https://arxiv.org/abs/2406.11288)                                                    | ICLR 2025 Workshop|
| RealFactBench | [RealFactBench: A Benchmark for Evaluating Large Language Models in Real-World Fact-Checking](https://www.arxiv.org/abs/2506.12538)                                       | arXiv 2025      |
| LongFact    | [Long-form factuality in large language models](https://arxiv.org/abs/2403.18802)                                                                                       | NeurIPS 2024    |
| PolitiHop   | [Multi-Hop Fact Checking of Political Claims](https://arxiv.org/abs/2009.06401)                                                                                         | IJCAI-2021      |
| FM2         | [Fool Me Twice: Entailment from Wikipedia Gamification](https://arxiv.org/abs/2104.04725)                                                                               | NAACL 2021      |
| HoVer       | [HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification](https://arxiv.org/abs/2011.03088)                                                                 | EMNLP 2020      |
| SCIFACT     | [Fact or fiction: Verifying scientific claims](https://arxiv.org/abs/2004.14974)                                                                                         | EMNLP 2020      |
| EX-FEVER    | [EX-FEVER: A Dataset for Multi-hop Explainable Fact Verification](https://arxiv.org/abs/2310.09754)                                                                       | ACL 2024        |
| FEVEROUS    | [FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information](https://arxiv.org/abs/2106.05707)                                               | NeurIPS 2021    |
| FactBench   | [FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation](https://arxiv.org/abs/2410.22257)                                                   | arXiv 2024      |

### Open-domain QA for Deep Research

| Name                 | Title                                                                                                                                                                      | Venue             |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| O2-QA                | [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582)                                             | arXiv 2025        |
| Researchy Questions  | [Researchy Questions: A Dataset of Multi-Perspective, Decompositional Questions for LLM Web Agents.](https://arxiv.org/abs/2402.17896)                                  | arXiv 2024        |
| MultimodalReportBench| [Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](https://arxiv.org/abs/2506.02454)                           | arXiv 2025        |
| DeepResearchGym      | [DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research](https://arxiv.org/abs/2505.19253)                                         | arXiv 2025        |
| Deep Research Bench  | [Deep Research Bench: Evaluating AI Web Research Agents](https://www.arxiv.org/abs/2506.06287)                                                                           | arXiv 2025        |
| DeepResearch Bench   | [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](https://arxiv.org/abs/2506.11763)                                                              | arXiv 2025        |
| WildSeek             | [Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations](https://arxiv.org/abs/2408.15232)                     | EMNLP 2024        |
| ProxyQA              | [PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models](https://arxiv.org/abs/2401.15042)                               | ACL 2024          |
| Long2RAG             | [Long2RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Recall](https://arxiv.org/abs/2410.23000)                                | EMNLP 2024        |
| ResearcherBench      | [ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of Scientific Inquiry](https://arxiv.org/abs/2507.16280)                                       | arXiv 2025        |
| ReportBench  |  [ReportBench: Evaluating Deep Research Agents via Academic Survey Tasks](http://arxiv.org/abs/2508.15804) | arXiv 2025  |
| DeepScholar-Bench  |	[DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis](https://arxiv.org/abs/2508.20033)  |  arXiv 2025 |
| ResearchQA  |  [ResearchQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics](http://arxiv.org/abs/2509.00496)  |  arXiv 2025 |
| DeepResearch Arena  | [DeepResearch Arena: The First Exam of LLMs' Research Abilities via Seminar-Grounded Tasks](https://arxiv.org/abs/2509.01396)  |  arXiv 2025 |
| DeepTRACE  |  [DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence](https://arxiv.org/abs/2509.04499)  |  arXiv 2025 |

### Domain-specific dataset

| Name            | Title                                                                                                                                                                      | Venue         |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| FinSearchBench-24| [An Agent Framework for Real-Time Financial Information Searching with Large Language Models](https://arxiv.org/abs/2502.15684)                                          | arXiv 2024    |
| MIRAGE          | [MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations](https://arxiv.org/abs/2506.20100)                     | arXiv 2025    |
| xbench          | [xbench: Tracking Agents Productivity Scaling with Profession-Aligned Real-World Evaluations](https://arxiv.org/abs/2506.13651)                                           | arXiv 2025    |
| SolutionBench   | [DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](https://arxiv.org/abs/2502.20730)                           | arXiv 2025    |
| DQA             | [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/abs/2406.12430)                           | NAACL 2024    |
| MedMCQA         | [MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering](https://arxiv.org/abs/2203.14371)                                     | PMLR 2022     |
| MedBrowseCom    | [MedBrowseComp: Benchmarking Medical Deep Research and Computer Use](https://arxiv.org/abs/2505.14963)                                                                   | arXiv 2025    |
| GPQA            | [Gpqa: A graduate-level google-proof q&a benchmark.](https://arxiv.org/abs/2311.12022)                                                                                   | COLM 2024     |
| ScIRGen-Geo     | [ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research](https://arxiv.org/abs/2506.11117)                                                    | KDD 2025      |
| OlympiadBench   | [Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems](https://arxiv.org/abs/2402.14008)               | ACL 2024      |
| DeepShop        | [DeepShop: A Benchmark for Deep Research Shopping Agents](https://arxiv.org/abs/2506.02839)                                                                              | arXiv 2025    |
| USACO           | [Can Language Models Solve Olympiad Programming?](https://arxiv.org/abs/2404.10952)                                                                                     | COLM 2024     |
| GAIA            | [GAIA: a benchmark for general AI assistants.](https://arxiv.org/abs/2311.12983)                                                                                         | arXiv 2023    |
| HLE             | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249)                                                                                                                | arXiv 2025    |
| HERB            | [Benchmarking Deep Search over Heterogeneous Enterprise Data](https://arxiv.org/abs/2506.23139)                                                                          | arXiv 2025    |
| FinAgentBench  | [FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering](https://arxiv.org/abs/2508.14052) | arXiv 2025 |

### Other Aspect

| Name         | Title                                                                                                                                                                      | Venue         |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| Instruct2DS    | [AutoData: A Multi-Agent System for Open Web Data Collection](https://arxiv.org/abs/2505.15859)                                                                           | arXiv 2025    |
| IIRC         | [IIRC: A Dataset of Incomplete Information Reading Comprehension Questions](https://arxiv.org/abs/2011.07127)                                                            | EMNLP 2020    |
| Search Arena | [Search Arena: Analyzing Search-Augmented LLMs](https://arxiv.org/abs/2506.05334)                                                                                        | arXiv 2025    |
| CONFLICTS    | [DRAGged into CONFLICTS:Detecting and Addressing Conflicting Sources in Search-Augmented LLMs](https://arxiv.org/abs/2506.08500)                                          | arXiv 2025    |
| WebWalkerQA  | [WebWalker: Benchmarking LLMs in Web Traversal](https://arxiv.org/abs/2501.07572)                                                                                         | arXiv 2025    |
| ToolQA       | [ToolQA: A Dataset for LLM Question Answering with External Tools](https://arxiv.org/abs/2306.13304)                                                                      | arXiv 2023    |
| RAGChecker   | [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/abs/2408.08067)                                                    | arXiv 2024    |
| DRComparator | [Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents](https://arxiv.org/abs/2507.05495)                                       | arXiv 2025    |
| RAVine       | [RAVine: Reality-Aligned Evaluation for Agentic Search](https://arxiv.org/abs/2507.16725)                                                                                 | arXiv 2025    |
| WideSearch   | [WideSearch: Benchmarking Agentic Broad Info-Seeking](https://arxiv.org/abs/2508.07999)  | arXiv 2025 | 


Feel free to open an issue or PR to add new papers and benchmarks!

