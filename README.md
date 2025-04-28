# ML_Assignment2

# *პროექტის მიზანი*
Kaggle-ის კონკურსი მიზნად ისახავს IEEE-ს დაუბალანსებელ დატასეტზე 400ზე მეტი ცვლადის გამოყენებით fraudulent ტრანზაქციების დეტექციას. პროექტში გატესტილია სამი მოდელი RandomForest, Adaboost და XGBoost.

# *რეპოზიტორიის სტრუქტურა*
model_experiment_rf.ipynb - RandomForest მოდელის სრული pipeline-ის აღწერა.
model_experiment_ada.ipynb - Adaboost მოდელის სრული pipeline-ის აღწერა.
model_experiment_xgb.ipynb - XGBoost მოდელის სრული pipeline-ის აღწერა.

model_inference_rf.ipynb - საუკეთესო RandomForest მოდელით ტესტ სეტზე პრედიქცია.
model_inference_ada.ipynb - საუკეთესო Adaboost მოდელით ტესტ სეტზე პრედიქცია.
model_inference_xgb.ipynb - საუკეთესო XGBoost მოდელით ტესტ სეტზე პრედიქცია.

rf_final.csv, ada_final.csv, xgb_final.csv - თითოეული model_inference-ით დაგენერირებული პრედიქციები ტესტ სეტზე.

# *Feature engineering*
NaN მნიშვნელობების შევსებისას რიცხვით ცვლადებში ჩაწერილია 0, ხოლო კატეგორიულ ცვლადებში no_{category_name} სტრინგი.

კატეგორიული ცვლადების რიცხვითად გადაყვანის ნაწილში გამოყენებულია WOE encoding, რადგან კატეგორიული ცვლადები 3+ მნიშვნელობებით გვხვდებიან. ამ ნაწილში ინტეგრირებულია Laplacian smoothing, რადგან ზოგიერთი ცვლადისთვის WOE მნიშვნელობის გამოთვლისას მნიშვნელში ნული ჯდება.

Cleaning ნაწილში მონაცემების წაკითხვისა და პირველადი ინსპექციის შემდეგ გადაგდებული მაქვს 6 ცვლადი, რომლებსაც ზედმეტად ბევრი განსხვავებული მნიშვნელობა ჰქონდათ და მოდელი მათზე კარგად ვერ ისწავლიდა. (bad_cols = ['id_33','id_31','id_30','P_emaildomain','R_emaildomain', 'DeviceInfo'])


# *Feature selection*
Feature selection ნაწილში გამოყენებულია კორელაციის ფილტრი 0.8-იანი threshold-ით, გადარჩეული ცვლადებიდან საუკეთესოებს კი Recursive Feature Selector ირჩევს.

# *Training*
პროექტში გატესტილია სამი მოდელი: RandomForest, XGBoost, Adaboost. ერთ-ერთი მთავარი პრობლემა, რასაც გადავეყარე, იყო ის, რომ GridSearch-ის გაშვების შემდეგ მონაცემთა დიდი რაოდენობის გამო Kaggle-ს მეხსიერება ივსებოდა და ნოუთბუქი იქრაშებოდა, ამიტომ ყველა ექსპერიმენტი ინდივიდუალურადაა გაშვებული. პარამეტრების tuning-ს ყოველი გაშვების შემდეგ precision და recall-ის ანალიზის საფუძველზე მცირედით ვცვლიდი და ოპტიმალური შედეგისკენ სვლას ვცდილობდი.

თითოეული მოდელის ტრენინგამდე ხდება მონაცემების Downsampling და 3.5%-იანი თარგეთის წილის 30%-მდე გაზრდა. თავდაპირველად XGBoost მოდელს Downsampling-ის გარეშე ვაკეთებდი, რის გამოც დაბალი recall ჰქონდა, რამდენიმე ექსპერიმენტის შემდეგ კი RandomUndersampler-ის 0.3-იანმა sampling_strategy პარამეტრმა ყველაზე კარგად გაამართლა და F1 score 0.6-ის მიდამოდან 0.8-მდე აიყვანა. საიტერესო იყო learning rate-ისა და ხეების რაოდენობის ბალანსის დაჭერაც ისე, რომ მოდელი overfit-ში არ გასულიყო. learning rate-ის ნელ-ნელა დაწევამ და მოდელის კომპლექსურობის ხეების რაოდენობის ხარჯზე გაზრდამ დროთა განმავლობაში უკეთესი შედეგი გამოიღო. XGBoost-ზე საუკეთესო პარამეტრების ხელით შერჩევის შემდეგ დანარჩენ ორ მოდელზე ოპტიმალური პარამეტრების შერჩევა რთული აღარ ყოფილა.

მოდელის სიძლიერის შესაფასებელ მთავარ მეტრიკებად გამოყენებულია Precision, recall და F1 score. თითოეულ ტრენინგ ექსპერიმენტში მოდელის პარამეტრებთან ერთად დალოგილია test/validation-ის f1, precision, recall, auroc, prediction threshold მაჩვენებლები. ამასთანავე, json არტიფაქტებადაა შენახული ფაიფლაინის ყოველი ეტაპის ინფორმაცია: woe_encoding მაჩვენებლები, rfe-ს გადარჩეული და კორელაციის ფილტრის მიერ დადროპილი ცვლადები. შენახულია auprc, auroc და confusion matrix პლოტებიც.

მოდელებმა შემდეგი Train/Validation სქორები დადეს:
RandomForest - F1 0.75/0.71
AdaBoost - F1 0.69/0.66
XGBoost - F1 0.81/0.77

# *MLFlow tracking*
ექსპერიმენტების სასაფლაო:
https://dagshub.com/dimna21/ML_Assignment2/experiments

დარეგისტრირებული მოდელები:

RandomForest: https://dagshub.com/dimna21/ML_Assignment2/models/RF_Final/1
AdaBoost: https://dagshub.com/dimna21/ML_Assignment2/models/ADA_Final/1
XGBoost: https://dagshub.com/dimna21/ML_Assignment2/models/XGB_Final/1

