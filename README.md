# Graphs Smart Contract Codes

dans `./src/code_representation.ipynb` c'est un brouillon rapide de transformation de code `.sol` en graphe interprétable par le modèle **GraphsCodeBERT**, en qql sorte on doit séparer notre code en plusieurs input (les commentaires et les noms de la classe et des fonctions, les interactions avec les variables (d'où elles viennent) grâce au **DataFlow** et pour finir la liste des variables) après ça on le passe dans le modèle (modèle basé sur Bret) qu'on aura finetuner juste avant pour qu'il nous dit clrmt si c'est fraude ou pas. Pour aller plus loin on peut aussi finetuner graphscodebert pour qu'il nous donne exactement le type de problème (reentracy, overflow etc...)

TODO pour la suite:  
- comprendre en détail le mécanisme de graphscodebert (cf `.refs/graphs-code-bert.pdf`)
- finetuner le modèle pour notre cas à nous
- mettre au propre un notebook de test
- tester notre modèle sur des vrais codes

discussion:
- tester peut-être avec codeBERT qui est du même style pour avoir plusieurs résultats puis ensuite prendre le meilleurs
- tester d'autres approches pour peut-être créer notre propre modèle (?)