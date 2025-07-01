Bonjour,

Je tiens à préciser que je me suis sentie un peu perdue au début du projet, notamment sur la compréhension globale du sujet et les outils à utiliser. Malgré mes efforts, le déploiement n’a pas pu être finalisé comme je l’aurais souhaité.

J’ai rencontré plusieurs difficultés, notamment liées à la taille du fichier du dataset CIFAR-10 (plus de 160 Mo), ce qui a bloqué :

Le déploiement sur Render et Railway (limite d’image Docker à 4 Go).
Ces blocages m’ont permis d’apprendre des éléments importants comme :

Séparer proprement le code des données.
Et mieux comprendre les contraintes de déploiement sur des plateformes cloud gratuites.
Je vous remercie pour votre compréhension. Ce projet m’a beaucoup appris, et je reste motivée à m’améliorer.

Salma Wadouachi

pour lancer le projet :
source env/bin/activate   
python3 app.py      

Cette application permet d’uploader une image et, grâce à un réseau de neurones convolutifs (CNN), de détecter et classifier ce que représente l’image parmi les catégories du dataset CIFAR-10
