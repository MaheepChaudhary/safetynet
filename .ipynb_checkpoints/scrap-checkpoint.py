#!/usr/bin/env python3
"""Explore and analyze AddSent dataset structure"""

import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any

def load_addsent(path: str) -> Dict:
    """Load AddSent dataset from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_basic_stats(data: Dict) -> Dict[str, Any]:
    """Get basic statistics about the dataset"""
    stats = {
        'version': data.get('version', 'unknown'),
        'num_articles': len(data.get('data', [])),
        'num_paragraphs': 0,
        'num_questions': 0,
        'num_answers': 0,
        'titles': []
    }
    
    for article in data.get('data', []):
        stats['titles'].append(article.get('title', 'No title'))
        stats['num_paragraphs'] += len(article.get('paragraphs', []))
        
        for paragraph in article.get('paragraphs', []):
            stats['num_questions'] += len(paragraph.get('qas', []))
            
            for qa in paragraph.get('qas', []):
                stats['num_answers'] += len(qa.get('answers', []))
    
    return stats

def analyze_questions(data: Dict) -> Dict[str, Any]:
    """Analyze question patterns"""
    question_starts = Counter()
    question_lengths = []
    all_questions = []
    
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                all_questions.append(question)
                question_lengths.append(len(question.split()))
                
                # Get first word
                first_word = question.split()[0] if question else ''
                question_starts[first_word] += 1
    
    return {
        'total_questions': len(all_questions),
        'avg_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        'min_length': min(question_lengths) if question_lengths else 0,
        'max_length': max(question_lengths) if question_lengths else 0,
        'common_starts': question_starts.most_common(10),
        'sample_questions': all_questions[:5]
    }

def analyze_contexts(data: Dict) -> Dict[str, Any]:
    """Analyze context patterns"""
    context_lengths = []
    all_contexts = []
    
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '')
            all_contexts.append(context)
            context_lengths.append(len(context.split()))
    
    return {
        'total_contexts': len(all_contexts),
        'avg_length': sum(context_lengths) / len(context_lengths) if context_lengths else 0,
        'min_length': min(context_lengths) if context_lengths else 0,
        'max_length': max(context_lengths) if context_lengths else 0,
        'sample_contexts': all_contexts[:3]
    }

def analyze_answers(data: Dict) -> Dict[str, Any]:
    """Analyze answer patterns"""
    answer_lengths = []
    answer_texts = []
    multi_answer_count = 0
    
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            for qa in paragraph.get('qas', []):
                answers = qa.get('answers', [])
                
                if len(answers) > 1:
                    multi_answer_count += 1
                
                for answer in answers:
                    text = answer.get('text', '')
                    answer_texts.append(text)
                    answer_lengths.append(len(text.split()))
    
    return {
        'total_answers': len(answer_texts),
        'avg_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        'min_length': min(answer_lengths) if answer_lengths else 0,
        'max_length': max(answer_lengths) if answer_lengths else 0,
        'multi_answer_questions': multi_answer_count,
        'sample_answers': answer_texts[:10]
    }

def get_sample_examples(data: Dict, n: int = 3) -> List[Dict]:
    """Get sample QA examples"""
    examples = []
    count = 0
    
    for article in data.get('data', []):
        if count >= n:
            break
            
        for paragraph in article.get('paragraphs', []):
            if count >= n:
                break
                
            context = paragraph.get('context', '')
            
            for qa in paragraph.get('qas', []):
                if count >= n:
                    break
                
                examples.append({
                    'title': article.get('title', 'No title'),
                    'question': qa.get('question', ''),
                    'id': qa.get('id', ''),
                    'answers': [a.get('text', '') for a in qa.get('answers', [])],
                    'context': context[:300] + '...' if len(context) > 300 else context
                })
                count += 1
    
    return examples

def detect_adversarial_patterns(data: Dict) -> Dict[str, Any]:
    """Detect potential adversarial patterns in AddSent"""
    suspicious_patterns = []
    pattern_counts = defaultdict(int)
    
    # Common adversarial indicators
    adversarial_keywords = [
        '7th law', 'eighth law', '9th law',  # Wrong law numbers
        'never', 'always', 'impossible',      # Absolute statements
        'however', 'although', 'but',         # Contradictions
    ]
    
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '').lower()
            
            for keyword in adversarial_keywords:
                if keyword in context:
                    pattern_counts[keyword] += 1
                    if len(suspicious_patterns) < 5:  # Keep first 5 examples
                        suspicious_patterns.append({
                            'keyword': keyword,
                            'context_snippet': context[max(0, context.find(keyword)-50):context.find(keyword)+100]
                        })
    
    return {
        'pattern_counts': dict(pattern_counts),
        'suspicious_examples': suspicious_patterns
    }

def print_report(data: Dict, args):
    """Print comprehensive analysis report"""
    print("="*80)
    print("ADDSENT DATASET ANALYSIS REPORT")
    print("="*80)
    
    # Basic stats
    print("\n📊 BASIC STATISTICS")
    print("-"*80)
    stats = get_basic_stats(data)
    print(f"Version: {stats['version']}")
    print(f"Number of articles: {stats['num_articles']}")
    print(f"Number of paragraphs: {stats['num_paragraphs']}")
    print(f"Number of questions: {stats['num_questions']}")
    print(f"Number of answers: {stats['num_answers']}")
    print(f"\nArticle titles: {', '.join(stats['titles'][:5])}")
    if len(stats['titles']) > 5:
        print(f"... and {len(stats['titles'])-5} more")
    
    # Question analysis
    print("\n❓ QUESTION ANALYSIS")
    print("-"*80)
    q_stats = analyze_questions(data)
    print(f"Total questions: {q_stats['total_questions']}")
    print(f"Average question length: {q_stats['avg_length']:.2f} words")
    print(f"Length range: {q_stats['min_length']} - {q_stats['max_length']} words")
    print(f"\nMost common question starts:")
    for word, count in q_stats['common_starts']:
        print(f"  {word}: {count}")
    
    if args.show_samples:
        print(f"\nSample questions:")
        for i, q in enumerate(q_stats['sample_questions'], 1):
            print(f"  {i}. {q}")
    
    # Context analysis
    print("\n📝 CONTEXT ANALYSIS")
    print("-"*80)
    c_stats = analyze_contexts(data)
    print(f"Total contexts: {c_stats['total_contexts']}")
    print(f"Average context length: {c_stats['avg_length']:.2f} words")
    print(f"Length range: {c_stats['min_length']} - {c_stats['max_length']} words")
    
    if args.show_samples:
        print(f"\nSample contexts:")
        for i, c in enumerate(c_stats['sample_contexts'], 1):
            print(f"\n  Context {i}:")
            print(f"  {c[:200]}...")
    
    # Answer analysis
    print("\n✅ ANSWER ANALYSIS")
    print("-"*80)
    a_stats = analyze_answers(data)
    print(f"Total answers: {a_stats['total_answers']}")
    print(f"Average answer length: {a_stats['avg_length']:.2f} words")
    print(f"Length range: {a_stats['min_length']} - {a_stats['max_length']} words")
    print(f"Questions with multiple answers: {a_stats['multi_answer_questions']}")
    
    if args.show_samples:
        print(f"\nSample answers:")
        for i, a in enumerate(a_stats['sample_answers'], 1):
            print(f"  {i}. {a}")
    
    # Adversarial patterns
    if args.detect_adversarial:
        print("\n🎯 ADVERSARIAL PATTERN DETECTION")
        print("-"*80)
        adv_stats = detect_adversarial_patterns(data)
        
        if adv_stats['pattern_counts']:
            print("Detected adversarial keywords:")
            for keyword, count in sorted(adv_stats['pattern_counts'].items(), 
                                        key=lambda x: x[1], reverse=True):
                print(f"  '{keyword}': {count} occurrences")
            
            print("\nSuspicious context examples:")
            for example in adv_stats['suspicious_examples']:
                print(f"\n  Keyword: '{example['keyword']}'")
                print(f"  Context: ...{example['context_snippet']}...")
        else:
            print("No obvious adversarial patterns detected with common keywords.")
    
    # Sample examples
    if args.show_examples:
        print("\n📋 COMPLETE SAMPLE EXAMPLES")
        print("-"*80)
        examples = get_sample_examples(data, n=args.num_examples)
        
        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"  Title: {ex['title']}")
            print(f"  ID: {ex['id']}")
            print(f"  Question: {ex['question']}")
            print(f"  Answers: {', '.join(ex['answers'])}")
            print(f"  Context: {ex['context']}")

def export_for_inspection(data: Dict, output_path: str, num_samples: int = 10):
    """Export sample examples to a readable text file"""
    examples = get_sample_examples(data, n=num_samples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ADDSENT DATASET SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(examples, 1):
            f.write(f"EXAMPLE {i}\n")
            f.write("-"*80 + "\n")
            f.write(f"Title: {ex['title']}\n")
            f.write(f"ID: {ex['id']}\n")
            f.write(f"Question: {ex['question']}\n")
            f.write(f"Answers: {', '.join(ex['answers'])}\n")
            f.write(f"\nContext:\n{ex['context']}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"Exported {num_samples} samples to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze AddSent dataset")
    parser.add_argument("--dataset_path", required=True,
                       help="Path to AddSent JSON file")
    parser.add_argument("--show_samples", action="store_true",
                       help="Show sample questions, contexts, and answers")
    parser.add_argument("--show_examples", action="store_true",
                       help="Show complete examples")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of complete examples to show")
    parser.add_argument("--detect_adversarial", action="store_true",
                       help="Detect adversarial patterns")
    parser.add_argument("--export", type=str, default=None,
                       help="Export samples to text file")
    parser.add_argument("--export_samples", type=int, default=10,
                       help="Number of samples to export")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    data = load_addsent(args.dataset_path)
    
    # Print report
    print_report(data, args)
    
    # Export if requested
    if args.export:
        export_for_inspection(data, args.export, args.export_samples)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
    
    
# python scrap.py --dataset_path utils/data/dataset/addsent.json --show_samples --show_examples --detect_adversarial --num_examples 5