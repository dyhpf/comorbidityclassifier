import os
from typing import Dict, List, Optional, Any, Tuple
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langgraph.graph import Graph, END
from langchain.schema import BaseMessage, HumanMessage
import json
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

class SleepDisorderClassifier:
    """Multi-agent classifier for sleep disorders using Mixtral 24B"""
    
    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.tokenizer, self.llm = self._load_model()
        
        # Initialize components
        self.top_level_agents = self._initialize_top_level_agents()
        self.subcategory_agents = self._initialize_subcategory_agents()
        self.result_judger = ResultJudger(self.llm)
        self.contradiction_solver = ContradictionSolver(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        # Metrics tracking
        self.metrics = {
            "total_tokens": 0,
            "contradictions_resolved": 0,
            "early_stops": 0
        }
    
    def _load_model(self):
        """Load Mixtral 24B model with quantization"""
        try:
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return tokenizer, llm
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to CPU if GPU memory insufficient
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return tokenizer, llm

class ClassificationOutputParser(BaseOutputParser):
    """Parser for sleep disorder classification outputs"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse agent output into structured format"""
        text = text.strip().lower()
        
        # Handle boolean responses
        if any(word in text for word in ["true", "yes", "present", "detected", "confirmed"]):
            return {"classification": True, "confidence": "high", "reasoning": text}
        elif any(word in text for word in ["false", "no", "absent", "not detected", "ruled out"]):
            return {"classification": False, "confidence": "high", "reasoning": text}
        
        # Handle condition lists for subtypes
        if "conditions:" in text.lower() or "[" in text:
            try:
                if "[" in text and "]" in text:
                    start = text.index("[")
                    end = text.index("]") + 1
                    conditions_str = text[start:end]
                    conditions = json.loads(conditions_str)
                    return {"classification": conditions, "type": "condition_list"}
            except:
                pass
        
        # Default to False if unclear
        return {"classification": False, "confidence": "low", "reasoning": text}

class BaseSleepDisorderAgent:
    """Base class for sleep disorder classification agents"""
    
    def __init__(self, name: str, system_prompt: str, llm):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_parser = ClassificationOutputParser()
        
    def classify(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform classification on input text"""
        prompt = self._build_prompt(text, context)
        
        try:
            response = self.llm.invoke(prompt)
            return self.output_parser.parse(response)
        except Exception as e:
            print(f"Error in classification: {e}")
            return {"classification": False, "confidence": "low", "reasoning": f"Error: {str(e)}"}
    
    def _build_prompt(self, text: str, context: Optional[Dict] = None) -> str:
        """Build the prompt for sleep disorder classification"""
        if context:
            context_str = f"\nParent Context: {context.get('decision', 'N/A')}\nEvidence: {context.get('evidence', 'N/A')}"
        else:
            context_str = ""
            
        prompt = f"""[INST] {self.system_prompt}

CLINICAL TEXT TO ANALYZE:
{text}{context_str}

Please analyze the text and provide your classification. Be precise and evidence-based.

Your response should be concise and focused on the classification decision. [/INST]"""
        
        return prompt

class TopLevelSleepAgent(BaseSleepDisorderAgent):
    """Top-level sleep disorder classification agent"""
    
    def __init__(self, disorder: str, llm):
        system_prompt = self._get_system_prompt(disorder)
        super().__init__(f"TopLevel_{disorder}", system_prompt, llm)
    
    def _get_system_prompt(self, disorder: str) -> str:
        """Get specialized prompt for each sleep disorder"""
        prompts = {
            "insomnia": """
            You are a sleep medicine specialist analyzing clinical text for Insomnia Disorder.
            Determine if there is evidence of chronic insomnia based on these criteria:
            - Difficulty initiating or maintaining sleep
            - Early morning awakening with inability to return to sleep
            - Sleep difficulties occurring despite adequate opportunity for sleep
            - Significant distress or impairment in daytime functioning
            - Symptoms present at least 3 nights per week for at least 3 months
            
            Return ONLY "True" if insomnia is present, "False" otherwise.
            Provide brief clinical reasoning.
            """,
            
            "sleep_apnea": """
            You are a sleep medicine specialist analyzing clinical text for Sleep Apnea.
            Determine if there is evidence of obstructive sleep apnea based on:
            - Witnessed apneas or breathing interruptions during sleep
            - Loud snoring, gasping, or choking sounds
            - Excessive daytime sleepiness, fatigue
            - Morning headaches, dry mouth
            - Obesity, large neck circumference, hypertension
            - Nocturnal restlessness, frequent awakenings
            
            Return ONLY "True" if sleep apnea is present, "False" otherwise.
            Provide brief clinical reasoning.
            """,
            
            "restless_legs": """
            You are a sleep medicine specialist analyzing clinical text for Restless Legs Syndrome (RLS).
            Determine if there is evidence of RLS based on:
            - Urge to move legs, usually accompanied by uncomfortable sensations
            - Symptoms begin or worsen during periods of rest or inactivity
            - Symptoms partially or totally relieved by movement
            - Symptoms worse in the evening or night
            - Symptoms not solely accounted for by another medical condition
            
            Return ONLY "True" if RLS is present, "False" otherwise.
            Provide brief clinical reasoning.
            """,
            
            "narcolepsy": """
            You are a sleep medicine specialist analyzing clinical text for Narcolepsy.
            Determine if there is evidence of narcolepsy based on:
            - Excessive daytime sleepiness with sleep attacks
            - Cataplexy (sudden loss of muscle tone)
            - Sleep paralysis, hypnagogic hallucinations
            - Disrupted nighttime sleep
            - Automatic behaviors
            
            Return ONLY "True" if narcolepsy is present, "False" otherwise.
            Provide brief clinical reasoning.
            """,
            
            "circadian_rhythm": """
            You are a sleep medicine specialist analyzing clinical text for Circadian Rhythm Sleep-Wake Disorders.
            Determine if there is evidence of circadian rhythm disorders based on:
            - Persistent misalignment between sleep-wake pattern and environmental demands
            - Delayed sleep phase (falling asleep very late)
            - Advanced sleep phase (falling asleep very early)
            - Irregular sleep-wake pattern
            - Shift work disorder symptoms
            
            Return ONLY "True" if circadian rhythm disorder is present, "False" otherwise.
            Provide brief clinical reasoning.
            """
        }
        
        return prompts.get(disorder, f"Analyze for {disorder} presence. Return True if present, False otherwise.")

class SubtypeSleepAgent(BaseSleepDisorderAgent):
    """Subtype classification agent for sleep disorders"""
    
    def __init__(self, parent_disorder: str, subtype: str, llm):
        system_prompt = self._get_subtype_prompt(parent_disorder, subtype)
        super().__init__(f"Subtype_{subtype}", system_prompt, llm)
    
    def _get_subtype_prompt(self, parent_disorder: str, subtype: str) -> str:
        """Get specialized prompt for sleep disorder subtypes"""
        subtype_prompts = {
            "insomnia": {
                "chronic_insomnia": """
                Analyze for Chronic Insomnia Disorder subtype.
                This requires:
                - Sleep difficulties at least 3 nights per week
                - Duration of at least 3 months
                - Significant daytime impairment
                - Not better explained by other sleep disorders
                """,
                "short_term_insomnia": """
                Analyze for Short-Term Insomnia subtype.
                This involves:
                - Sleep difficulties present less than 3 months
                - Significant distress or impairment
                - Often related to specific stressors
                """
            },
            "sleep_apnea": {
                "mild_osa": """
                Analyze for Mild Obstructive Sleep Apnea.
                Look for evidence of:
                - AHI 5-15 events per hour
                - Mild symptoms, may not require CPAP
                - Positional or mild cases
                """,
                "moderate_osa": """
                Analyze for Moderate Obstructive Sleep Apnea.
                Look for evidence of:
                - AHI 15-30 events per hour
                - Moderate daytime sleepiness
                - Often requires treatment
                """,
                "severe_osa": """
                Analyze for Severe Obstructive Sleep Apnea.
                Look for evidence of:
                - AHI >30 events per hour
                - Severe daytime impairment
                - Often with comorbidities
                """
            },
            "restless_legs": {
                "intermittent_rls": """
                Analyze for Intermittent RLS subtype.
                Characteristics:
                - Symptoms occur less than twice weekly
                - Mild to moderate distress
                """,
                "chronic_rls": """
                Analyze for Chronic Persistent RLS subtype.
                Characteristics:
                - Symptoms occur at least twice weekly
                - Moderate to severe distress
                """
            }
        }
        
        parent_prompts = subtype_prompts.get(parent_disorder, {})
        return parent_prompts.get(subtype, f"Analyze for {subtype} presence. Return True if present, False otherwise.")

class ResultJudger:
    """Early judgment/filtering node for sleep disorders"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def judge(self, text: str, classification_result: Dict, agent_name: str) -> Dict[str, Any]:
        """Judge if the classification result is reliable enough to proceed"""
        prompt = f"""[INST] Assess the reliability of this sleep disorder classification:

Agent: {agent_name}
Clinical Text: {text}
Classification: {classification_result.get('classification')}
Reasoning: {classification_result.get('reasoning', 'N/A')}

Is this classification reliable enough to proceed with subtype analysis?
Consider: evidence quality, confidence, clinical relevance.

Return ONLY "proceed" or "stop" with brief reasoning. [/INST]"""
        
        try:
            response = self.llm.invoke(prompt)
            decision = "proceed" if "proceed" in response.lower() else "stop"
            
            return {
                "decision": decision,
                "original_result": classification_result,
                "judgment_reasoning": response
            }
        except Exception as e:
            return {
                "decision": "proceed",  # Default to proceed on error
                "original_result": classification_result,
                "judgment_reasoning": f"Error in judgment: {str(e)}"
            }

class ContradictionSolver:
    """Reconcile parent-child contradictions for sleep disorders"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def solve_contradiction(self, parent_result: Dict, child_results: List[Dict], 
                          clinical_text: str) -> Dict[str, Any]:
        """Solve contradictions between parent and child classifications"""
        
        parent_class = parent_result.get('classification', False)
        child_any_true = any(result.get('classification', False) for result in child_results)
        
        # No contradiction if both agree
        if parent_class == child_any_true:
            return {
                "resolved": True,
                "final_parent": parent_class,
                "final_children": child_results,
                "contradiction": False,
                "reasoning": "No contradiction found"
            }
        
        # Contradiction detected - resolve using Mixtral
        prompt = f"""[INST] Resolve sleep disorder classification contradiction:

CLINICAL TEXT: {clinical_text}

PARENT CLASSIFICATION: {parent_class}
Parent Reasoning: {parent_result.get('reasoning', 'N/A')}

CHILD RESULTS: 
{chr(10).join([f"• {r.get('reasoning', 'N/A')}" for r in child_results])}

Contradiction: Parent says {parent_class} but subtypes suggest {child_any_true}.

Re-analyze the clinical text and determine the correct classification.
Consider: clinical context, specificity of evidence, diagnostic criteria.

Return JSON format: {{"resolved_parent": bool, "reasoning": "clinical reasoning"}} [/INST]"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.index("}") + 1
                json_str = response[start:end]
                resolution = json.loads(json_str)
            else:
                # Fallback parsing
                resolution = {"resolved_parent": parent_class, "reasoning": "Fallback to parent"}
                
            return {
                "resolved": True,
                "final_parent": resolution.get("resolved_parent", parent_class),
                "final_children": child_results,
                "contradiction": True,
                "reasoning": resolution.get("reasoning", "Resolution applied")
            }
            
        except Exception as e:
            # Fallback: trust the parent
            return {
                "resolved": True,
                "final_parent": parent_class,
                "final_children": child_results,
                "contradiction": True,
                "reasoning": f"Error in resolution: {str(e)}"
            }

# Main Sleep Disorder Classifier Implementation
class SleepDisorderClassifier:
    """Multi-agent classifier for sleep disorders using Mixtral 24B"""
    
    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.tokenizer, self.llm = self._load_model()
        
        # Initialize components
        self.top_level_agents = self._initialize_top_level_agents()
        self.subcategory_agents = self._initialize_subcategory_agents()
        self.result_judger = ResultJudger(self.llm)
        self.contradiction_solver = ContradictionSolver(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        # Metrics tracking
        self.metrics = {
            "total_tokens": 0,
            "contradictions_resolved": 0,
            "early_stops": 0
        }
    
    def _load_model(self):
        """Load Mixtral model with optimized settings"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use quantization if GPU available
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float32,
                )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return tokenizer, llm
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _initialize_top_level_agents(self) -> Dict[str, TopLevelSleepAgent]:
        """Initialize top-level sleep disorder agents"""
        disorders = [
            "insomnia",
            "sleep_apnea", 
            "restless_legs",
            "narcolepsy",
            "circadian_rhythm"
        ]
        
        return {disorder: TopLevelSleepAgent(disorder, self.llm) for disorder in disorders}
    
    def _initialize_subcategory_agents(self) -> Dict[str, List[SubtypeSleepAgent]]:
        """Initialize subtype agents organized by parent disorder"""
        subtypes = {
            "insomnia": [
                "chronic_insomnia",
                "short_term_insomnia"
            ],
            "sleep_apnea": [
                "mild_osa",
                "moderate_osa", 
                "severe_osa"
            ],
            "restless_legs": [
                "intermittent_rls",
                "chronic_rls"
            ]
        }
        
        return {
            parent: [SubtypeSleepAgent(parent, subtype, self.llm) for subtype in subs]
            for parent, subs in subtypes.items()
        }
    
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow for sleep disorder classification"""
        workflow = Graph()
        
        # Define nodes
        workflow.add_node("top_level_classification", self._top_level_classification)
        workflow.add_node("result_judgment", self._result_judgment)
        workflow.add_node("subtype_classification", self._subtype_classification)
        workflow.add_node("contradiction_resolution", self._contradiction_resolution)
        workflow.add_node("final_aggregation", self._final_aggregation)
        
        # Define edges
        workflow.set_entry_point("top_level_classification")
        workflow.add_edge("top_level_classification", "result_judgment")
        workflow.add_conditional_edges(
            "result_judgment",
            self._should_proceed_to_subtypes,
            {
                "proceed": "subtype_classification",
                "stop": "final_aggregation"
            }
        )
        workflow.add_edge("subtype_classification", "contradiction_resolution")
        workflow.add_edge("contradiction_resolution", "final_aggregation")
        workflow.add_edge("final_aggregation", END)
        
        return workflow
    
    def _top_level_classification(self, state: Dict) -> Dict:
        """Perform top-level classification for all sleep disorders"""
        clinical_text = state["clinical_text"]
        results = {}
        
        print("Performing top-level sleep disorder classification...")
        
        for disorder, agent in self.top_level_agents.items():
            result = agent.classify(clinical_text)
            results[disorder] = result
            
            # Track tokens
            self.metrics["total_tokens"] += len(clinical_text) // 4
        
        state["top_level_results"] = results
        return state
    
    def _result_judgment(self, state: Dict) -> Dict:
        """Apply early judgment/filtering"""
        clinical_text = state["clinical_text"]
        judgments = {}
        
        for disorder, result in state["top_level_results"].items():
            judgment = self.result_judger.judge(clinical_text, result, disorder)
            judgments[disorder] = judgment
            
            if judgment["decision"] == "stop":
                self.metrics["early_stops"] += 1
                print(f"Early stop for {disorder}")
        
        state["judgments"] = judgments
        return state
    
    def _should_proceed_to_subtypes(self, state: Dict) -> str:
        """Determine if we should proceed to subtype classification"""
        judgments = state["judgments"]
        
        for disorder, judgment in judgments.items():
            if (judgment["decision"] == "proceed" and 
                judgment["original_result"].get("classification", False) and
                disorder in self.subcategory_agents):
                return "proceed"
        
        return "stop"
    
    def _subtype_classification(self, state: Dict) -> Dict:
        """Perform subtype classification for relevant disorders"""
        clinical_text = state["clinical_text"]
        judgments = state["judgments"]
        subtype_results = {}
        
        print("Performing subtype classification...")
        
        for disorder, judgment in judgments.items():
            if (judgment["decision"] == "proceed" and 
                judgment["original_result"].get("classification", False) and
                disorder in self.subcategory_agents):
                
                # Forward parent context to children
                parent_context = {
                    "decision": judgment["original_result"].get("classification"),
                    "evidence": judgment["original_result"].get("reasoning", "")
                }
                
                disorder_results = []
                for subtype_agent in self.subcategory_agents[disorder]:
                    result = subtype_agent.classify(clinical_text, parent_context)
                    disorder_results.append({
                        "subtype": subtype_agent.name,
                        "result": result
                    })
                    
                    self.metrics["total_tokens"] += len(clinical_text) // 4
                
                subtype_results[disorder] = disorder_results
        
        state["subtype_results"] = subtype_results
        return state
    
    def _contradiction_resolution(self, state: Dict) -> Dict:
        """Resolve parent-child contradictions"""
        clinical_text = state["clinical_text"]
        resolved_results = {}
        
        for disorder, sub_results in state.get("subtype_results", {}).items():
            parent_result = state["top_level_results"][disorder]
            child_results = [r["result"] for r in sub_results]
            
            resolution = self.contradiction_solver.solve_contradiction(
                parent_result, child_results, clinical_text
            )
            
            if resolution["contradiction"] and resolution["resolved"]:
                self.metrics["contradictions_resolved"] += 1
                print(f"Resolved contradiction for {disorder}")
            
            resolved_results[disorder] = resolution
        
        state["resolved_results"] = resolved_results
        return state
    
    def _final_aggregation(self, state: Dict) -> Dict:
        """Aggregate final results"""
        final_results = {
            "top_level": {},
            "subtypes": {},
            "metrics": self.metrics.copy()
        }
        
        # Top-level results (after judgment)
        for disorder, judgment in state.get("judgments", {}).items():
            final_results["top_level"][disorder] = {
                "classification": judgment["original_result"].get("classification", False),
                "reasoning": judgment["original_result"].get("reasoning", ""),
                "judgment": judgment["decision"],
                "judgment_reasoning": judgment["judgment_reasoning"]
            }
        
        # Subtype results (after resolution)
        for disorder, resolution in state.get("resolved_results", {}).items():
            final_results["subtypes"][disorder] = {
                "parent_final": resolution["final_parent"],
                "children": [
                    {
                        "subtype": state["subtype_results"][disorder][i]["subtype"],
                        "classification": child.get("classification", False),
                        "reasoning": child.get("reasoning", "")
                    }
                    for i, child in enumerate(resolution["final_children"])
                ],
                "had_contradiction": resolution["contradiction"],
                "resolution_reasoning": resolution["reasoning"]
            }
        
        # Calculate consistency metrics
        final_results["consistency_metrics"] = self._calculate_consistency_metrics(state)
        
        state["final_results"] = final_results
        return state
    
    def _calculate_consistency_metrics(self, state: Dict) -> Dict:
        """Calculate hierarchical consistency metrics"""
        hcr_count = 0
        total_comparisons = 0
        
        for disorder, resolution in state.get("resolved_results", {}).items():
            parent_final = resolution["final_parent"]
            child_any_true = any(child.get("classification", False) 
                               for child in resolution["final_children"])
            
            if parent_final == child_any_true:
                hcr_count += 1
            total_comparisons += 1
        
        hcr = hcr_count / total_comparisons if total_comparisons > 0 else 1.0
        
        return {
            "hierarchical_consistency_rate": hcr,
            "total_contradictions_resolved": self.metrics["contradictions_resolved"],
            "early_stops": self.metrics["early_stops"],
            "estimated_tokens": self.metrics["total_tokens"]
        }
    
    def classify(self, clinical_text: str) -> Dict[str, Any]:
        """Main classification method"""
        print("Starting sleep disorder classification...")
        initial_state = {
            "clinical_text": clinical_text,
            "timestamp": datetime.now().isoformat()
        }
        
        result = self.app.invoke(initial_state)
        print("Classification completed!")
        return result["final_results"]

# Example usage and evaluation
def evaluate_sleep_classifier():
    """Evaluate the sleep disorder classifier with example texts"""
    
    # Initialize classifier
    classifier = SleepDisorderClassifier()
    
    test_cases = [
        "Patient presents with chronic difficulty falling asleep, taking over 2 hours to fall asleep 4-5 nights per week for the past 6 months. Reports daytime fatigue and irritability.",
        "45-year-old male with loud snoring, witnessed apneas by spouse, and excessive daytime sleepiness. BMI 32, neck circumference 17 inches.",
        "Patient describes uncomfortable creeping sensations in legs in evenings, relieved by movement. Symptoms occur 3-4 times weekly for past year.",
        "Young adult with sudden sleep attacks during daytime, occasional muscle weakness when laughing. Reports sleep paralysis episodes.",
        "Shift worker with rotating schedule complains of inability to sleep during day, constant fatigue, and difficulty staying awake during night shifts."
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*60}")
        print(f"Text: {text}")
        
        results = classifier.classify(text)
        
        # Print top-level results
        print("\nTOP-LEVEL SLEEP DISORDER RESULTS:")
        for disorder, result in results["top_level"].items():
            status = "ok" if result["classification"] else "false"
            print(f"  {status} {disorder.replace('_', ' ').title()}: {result['classification']}")
            if result["classification"]:
                print(f"     Reasoning: {result['reasoning'][:100]}...")
        
        # Print subtype results
        if results["subtypes"]:
            print("\nSUBTYPE RESULTS:")
            for disorder, sub_result in results["subtypes"].items():
                print(f"  {disorder.replace('_', ' ').title()}:")
                print(f"    Parent Final: {sub_result['parent_final']}")
                for child in sub_result['children']:
                    status = "ok" if child["classification"] else "false"
                    print(f"    {status} {child['subtype'].replace('_', ' ').title()}: {child['classification']}")
        
        # Print metrics
        metrics = results["consistency_metrics"]
        print(f"\nMETRICS:")
        print(f"  Hierarchical Consistency Rate: {metrics['hierarchical_consistency_rate']:.3f}")
        print(f"  Contradictions Resolved: {metrics['total_contradictions_resolved']}")
        print(f"  Early Stops: {metrics['early_stops']}")
        print(f"  Estimated Tokens: {metrics['estimated_tokens']}")

if __name__ == "__main__":
    # Run evaluation
    evaluate_sleep_classifier()
