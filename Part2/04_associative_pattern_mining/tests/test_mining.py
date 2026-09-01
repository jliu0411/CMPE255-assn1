from src.mining import frequent_itemsets,association_rules
def test_known_rule():
    baskets=[{'A','B'},{'A','B'},{'A','B'},{'A','C'},{'C'}];sets,n=frequent_itemsets(baskets,.2);rules=association_rules(sets,n,.5,1);rule=next(r for r in rules if r['antecedents']==['B'] and r['consequents']==['A']);assert rule['confidence']==1;assert rule['lift']>1
