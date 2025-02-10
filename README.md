La segmentazione delle immagini è uno degli argomenti portanti della Computer Vision le cui applicazioni spaziano dalla guida autonoma all’analisi medica. Questo processo consiste 
nel dividere un’immagine in più parti o regioni, ciascuna delle quali rappresenta un oggetto o un’area di interesse specifica. Tra le reti neurali convoluzionali, l’architettura 
UNet si è affermata come una delle più efficaci per la segmentazione semantica, grazie alla sua capacità di catturare dettagli sia globali che locali.
In questo progetto è stata affrontata la problematica della segmentazione delle immagini utilizzando un approccio basato su deep learning. E stato creato un dataset personalizzato di
immagini e le rispettive maschere di segmentazione, sfruttando l’ambiente di simulazione NVIDIA Omniverse. La piattaforma Omniverse ha permesso di generare un dataset sintetico di 
2000 immagini, accompagnate dalle relative maschere segmentate, garantendo un controllo accurato sui dati e una vasta varietà di scenari, oggetti ed illuminazione.
Il dataset così ottenuto è stato utilizzato per addestrare una rete UNet, con un backbone ResNet34 per consentire una migliore estrazione delle caratteristiche. Dopo l’addestramento,
la rete è stata testata su nuove immagini non presenti nel dataset di addestramento, per valutare la capacità di generalizzazione del modello.
