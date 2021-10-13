# Analyse des données

## Histogrammes des variables

La fonction *initialize_dataframe* décompresse un dataset au format lz4 à l'adresse du chemin qu'elle prend en entrée, puis le renvoie en sortie sous la forme d'un dataframe pandas. Les datasets train et test sont décompressés et stockés dans deux dataframes.

La fonction *first_look_at_the_datasets* permet de se faire une première idée de la structure des dataframes *df_train* et *df_test*.

Je note que le dataset train a 632'000 points, tandis que le dataset test en a 32'000.

J'ai ensuite mené une analyse du dataset train en commençant par faire un histogram de la distribution de chaque variable sur le dataset d'entraînement df_train. Pour cela j'ai appelé la fonction *histogram_plot* sur le dataframe *df_train*. Cette fonction sauvegarde un histogramme par variable numérique dans le dossier histograms_numerical et un histogramme par variable catégorique dans le dossier histograms_categorical. J'ai observé les résultats suivants :

- la variable demand prends des valeurs entre 0 et 192. Il y'a peu d'échantillons pour lesquels la demande est elevée, le nombre d'échantillon décroit exponentiellement 
avec la valeur de la demande. On aura donc peu d'échantillons pour apprendre quand est ce que la demande est élévée, il faudra donc faire attention au
surapprentissage et utiliser des techniques de régularisation.

- la variable price prends des valeurs entre 11 et 116. Les trains ont peu souvent des prix élevés. Le nombre d'échantillons décroit exponentiellement avec le prix.
L'algorithme aura peu d'exemples pour comprendre comment la demande réagit à un prix élévé. Il faudra donc utiliser des techniques de régularisation.

- la variable sale_day_x prends des valeurs entre -89 et -1. La répartition de cette variable sur l'interval dans lequel elle prend ses valeurs est assez équitable.
Il y'a un peu plus d'échantillons pour les valeurs hautes.

- la variable od_origin_month possède bien plus d'échantillon en septembre qu'en avril. Seul 1 pourcent du dataset concerne des trains partant en avril, là encore 
on peut avoir des problème de suraprentissage pour les trains d'avril.

- sur l'histogram de la variable sale_month, on remarque qu'il y'a bien plus de ventes en juillet qu'en mars, 
là encore on peut avoir des problèmes de surapprentissage.

- il y'a très peu de ventes en 2018, ce qui vient du fait je pense, que le dataset concerne toutes les ventes de billet de trains partant en 2019 et 2020.
C'est donc normal qu'il y'ait quelques achats en fin 2018, de billets pour des trains partant au début de 2019.

- la variable od_number_of_similar_2_hours prends des valeurs entre -1 et 4, od_number_of_similar_4_hours entre 0 et 7,
od_number_of_similar_12_hours entre 0 et 10.

- On voit que sur les colonnes od_number_of_similar_2_hours et od_number_of_similar_4_hours il n'y a quasiment pas de données manquantes (valeur = -1)
contrairement à od_number_of_similar_12_hours où il y'a environ 16'000 données manquantes sur les 632'000, soit  : 
cette feature manque a 2,5 pourcent des points du dataset, cela peut perturber l'apprentissage je vais donc enlever les points pour lesquels
od_number_of_similar_12_hours = -1.

- La variable destination_current_public_holiday est constante égale à 0, on peut donc la supprimer car elle n'apporte aucune information sur la demande.

- Il y'a plus d'échantillons pour des trains en provenance et à destination de "bb" et "cgm". Le minimum
d'échantillons est atteint pour les trains en provenance et à destination de "cdm" mais le nombre d'échantillons reste acceptable (>4%).

- destination_days_to_next_public_holiday et origin_days_to_next_public_holiday sont à peu de choses près les même variables, je décide donc
de supprimer origin_days_to_next_public_holiday. Cependant pour destination_days_to_next_school_holiday et origin_days_to_next_public_holiday,
il y'a quelques différences dues, je penses aux différentes zones scolaires (A,B,C) donc je vais garder les deux variables, car elles apportent 
toutes les deux une information intéréssante pour prédire la demande.

- la variable sale_week a des valeurs entre 0 et 6 ce qui ne correspond pas au numéro de semaine dans l'année. De plus sale_week est égale à sale_weekday sur tout 
le dataset, c'est pour ca que je vais retirer cette variable du dataset

- les autres variables ont une distribution dont les bornes méritent peu ou pas de commentaire 
(par exemple : sale_weekday prends des valeurs entre 0 et 6, les échantillons sont répartis de manière à peu près équitables entre 
les jours de la semaine, ce qui n'est pas étonnant).

L'analyse des histogrammes m'a mené à supprimer certaines colonnes et rangées des datasets train et test. 

## Nuages de points pour déceler des tendances entre la demande et les variables numériques

Je veux observer comment réagit la demande quand varient certaines variables numériques. Pour cela j'ai utilisé la fonction *scatter_plot* qui sauvegarde dans le dossier scatter_plots une nuage de point par variable numérique. J'ai observés les résultats suivants :

- demand vs od_origin_weekday : pas grand chose de flagrant, outre que je m'attendais à ce que la demande soit plus forte le vendredi et le dimanche.
Mais peut être que cela est du aux prix qui étaient plus élevés ces jours là.

- demand vs od_number_of_similar_X_hours : pour chacune de ces 3 variables (X = 2,4,12) la demande fluctue en fonction de od_number_of_similar_X_hours.
Ces variables seront probablement intéressantes pour prédire la demande.

- demand vs price : la demande est plus forte quand le prix est bas, sauf pour les prix extrêmement bas où la demande est très faible.
Ceci peut être expliqué j'imagine par le fait que les prix ont été très fortement baissé car personne ne voulait de ces billets de train.
Cette variable est sûrement avec sale_day_x la plus intéressante pour prédire la demande.

- demand vs sale_day_x : la demande augmente (allure quadratic voir exponentielle) avec la variable sale_day_x. Les plus fortes demandes sont toujours observées juste
avant le départ du train. 

- demand vs destination_days_to_next_school_holiday et origin_days_to_next_school_holiday: pour x > 50 la demande est beaucoup plus faible.

- demand vs od_destination time, od_origin_time, od_travel_time_minutes : la demande varie beaucoup en fonction de ces deux variables, 
à vu d'oeil ces nuages de points semblent ce comporter différemment en fonction de plusieurs cas 
(différents comportements sur différents intervals de x)
Cela me fait penser qu'un arbre de décision pourrait être un modèle efficace pour prédire la demande.

- demand vs sale_day : la demande diminue légèrement quand sale_day augmente.

- demand vs destination_days_to_next_public_holiday : la demande diminue légèrement quand destination_days_to_next_public_holiday augmente.

## Boxplots pour observer l'influence des variables catégoriques sur la demande

Les Boxplots permettent de lire rapidement des informations sur la répartition statistique de la demande pour chaque valeur d'une variable catégorique. On peut y lire le premier et le troisième quartile (respectivement le bas et le haut de la boîte). Le trait horizontal au-dessus de chaque boîte représente la valeur du troisième quartile plus une fois et demi l'écart interquartile (le troisième quartile moins le premier quartile).
Pour chaque box plot, les catégories sont triées par ordre croissant de la valeur de la médiane de la demande, avec à droite les catégories pour lesquelles la médiane est la plus haute.

Pour analyser l'influence des variables catégoriques sur la demande, j'ai utilisé la fonction *box_plot* qui sauvegarde dans le dossier box_plots une boxplot par variable catégorique. J'ai observés les résultats suivants :

- demand vs destination_station_name : les trains à destination de "bb" et "cpe" correspondent aux cas de plus fortes demandes.
Cependant dans le cas de "bb" les points de plus hautes demande sont bien largement supérieurs au troisième quartile + 1,5* l'écart interquartile.
Ceci est moins vrai pour les trains à destination de "cpe". La variable destination_station_name semble intéressante pour prédire la demande.

- demand vs origin_station_name : les trains en provenance de "rb" et "ag" correspondent aux cas de plus forte demande.
La variable origin_station_name semble intéressante pour prédire la demande.

- demand vs od_origin_current_public_holiday, od_origin_current_school_holiday, destination_current_school_holiday :
La demande semble être plus forte en dehors des vacances, qu'elles soient publiques ou scolaires, et que cela concerne
la destination ou le lieu de départ du train en question.

- demand vs od_origin_month/week/weekday : il y'a des fluctuations intéressantes de la demande pour chaques valeurs, qui 
je pense, pourraient plus facilement être comprises par un arbre de décision que par une régression linéaire.

- demand vs od_origin_year et sale_year : a vu d'oeil, on ne dirait pas que la demande fluctue par rapport à ces variables.

- demand vs sale_weekday : la demande varie un peu en fonction du jour de vente du billet de train dans la semaine

- demand vs sale_month : on remarque que la demande fluctue beaucoup en fonction du mois où les billets de train sont proposés.
Notamment on voit que le maximum, la médiane, le troisième quartile et le troisième quartile + 1.5*l'écart intercartile de la demande
varient beaucoup en fonction de sale_month. Ces variations pourront probablement être comprises par un arbre de décision.

L'analyse de l'influence des variables catégoriques et numériques sur la demande m'a mené au choix de modèles dont les briques de bases sont des arbres de décision.

# Entraînement et validation du modèle

J'ai choisi de tester les deux modèles suivants : 
- random forest
- gradient boosted trees

Ces deux modèles sont des modèles d'ensemble learning : technique qui consiste à utiliser une combinaison de modèle simples appelés "weak learners" pour accroître les performances du modèle global. 
Dans le cas de random forest et gradient boosted trees, les "weak learners' sont des arbres de décision.

Avant d'entraîner les deux modèles j'ai d'abord séparé le dataset train en deux datasets : 
- *df_train_partial* sert à l'apprentissage pour chaque modèle
- *df_valid* est le dataset de validation, il sert à mesurer les performances de chaque modèle. Comme les données sont inscrites dans le temps, j'ai pensé qu'il était important de valider les modèles sur des données postérieures à celles utilisées pendant l'entraînement, car c'est ce qui se passe en pratique.

J'ai ensuite codé la fonction *evaluate_model* qui prend en entrée un dataset concaténé avec les prédictions d'un modèle sur ce dataset, et print une table contenant les moyennes de l'erreur absolue et l'erreur relative sur la demande cumulée pour chaque paire de gare (origine,destination), et ce, pour plusieurs valeurs de nombre de jours cumulés. Cette fonction print aussi la valeur d'une métrique personnalisée : *custom_metric*, qui est la moyenne de l'erreur absolue coefficientée par les prix du billet. J'ai pensé que cette métrique permettrait de donner plus d'importance aux erreurs commises sur les billets chers. La fonction *evaluate_model* print la valeur de cette métrique personnalisée pour chaque paire de gare (origine,destination) dans une table.
La fonction calcule ensuite cette métrique personnalisée sur tout le dataset (en moyennant sur toute les destinations/origines)

J'ai ensuite utilisé la fonction *train_model_and_compute_results_on_validation_set* pour tester les deux modèles sur le dataset validation *df_valid*, et ce pour différentes valeurs des hyperparamètres :
- *ntrees* (nombre de "weak learners") 
- *max_depth* (profondeur des "weak learners")
J'ai reglé ces hyperparamètres dans le but d'optimiser la métrique personnalisée 

Les meilleurs résultats sont obtenu pour le modèle gradient boosted trees avec *ntrees = 100* et *max_depth = 5*.

 J'ai ensuite codé la fonction *train_selected_model_and_compute_results_on_test_set* qui entraîne le modèle choisi sur le dataset train entier et évalue ses résultats sur le dataset test. Voici mes observations concernant les résultats du modèle retenu :
- 20 jours avant le départ le modèle fait de grosses erreur en absolu cumulé sur les trains de "ag" vers "cpe" et de "cpe" vers "ag" puis plus on se rapproche du départ plus le modèle fait aussi beaucoup d'erreur absolu sur les trains de "rb" vers "bb" et de "bb" vers "rb".
- le modèle fait peu d'erreur en absolu sur les trains de "bb" vers "cgm" et de "cgm" vers "bb". Mais fait en revance le modèle fait beacoup d'erreur relative sur ces trains, ce qui est probablement explicable par le fait que la demande est plus faible pour ces trains. 
- concernant la métrique personnalisée, on observe à peu près les mêmes tendances que pour l'erreur cumulée absolue. Les résultats sont bons pour les trains de "bb" vers "cgm" et de "cgm" vers "bb". Pour les autres trains, les résultats sont moins bons et à peu près similaires entre eux.



