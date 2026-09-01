from __future__ import annotations
from collections import Counter
from itertools import combinations
import math

def frequent_itemsets(baskets:list[set[str]],min_support=.02,max_len=3,max_items=120):
    n=len(baskets);counts=Counter(x for b in baskets for x in b);floor=max(2,math.ceil(n*min_support));common={x for x,c in counts.most_common(max_items) if c>=floor};trimmed=[sorted(b&common) for b in baskets];result={frozenset([x]):c for x,c in counts.items() if x in common and c>=floor}
    pair_counts=Counter(pair for b in trimmed for pair in combinations(b,2));result.update({frozenset(k):v for k,v in pair_counts.items() if v>=floor})
    if max_len>=3:
        valid_pairs=set(k for k in result if len(k)==2);triple_counts=Counter()
        for b in trimmed:
            for tri in combinations(b,3):
                f=frozenset(tri)
                if all(frozenset(p) in valid_pairs for p in combinations(tri,2)):triple_counts[f]+=1
        result.update({k:v for k,v in triple_counts.items() if v>=floor})
    return result,n

def association_rules(itemsets:dict,n:int,min_confidence=.3,min_lift=1.1):
    rules=[]
    for itemset,count in itemsets.items():
        if len(itemset)<2:continue
        for size in range(1,len(itemset)):
            for ant_tuple in combinations(sorted(itemset),size):
                ant=frozenset(ant_tuple);con=itemset-ant
                if ant not in itemsets or con not in itemsets:continue
                support=count/n;confidence=count/itemsets[ant];con_support=itemsets[con]/n;lift=confidence/con_support
                if confidence<min_confidence or lift<min_lift:continue
                leverage=support-(itemsets[ant]/n)*con_support;conviction=(1-con_support)/(1-confidence) if confidence<1 else 999.0
                rules.append({'antecedents':sorted(ant),'consequents':sorted(con),'support':round(support,5),'confidence':round(confidence,5),'lift':round(lift,4),'leverage':round(leverage,5),'conviction':round(conviction,3),'count':count})
    return sorted(rules,key=lambda r:(r['lift']*r['support'],r['confidence']),reverse=True)

def rule_confidence(rule,baskets):
    a,c=set(rule['antecedents']),set(rule['consequents']);eligible=sum(a<=b for b in baskets);hits=sum(a<=b and c<=b for b in baskets);return hits/eligible if eligible else 0,eligible
