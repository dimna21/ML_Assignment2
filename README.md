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
პროექტში გატესტილია სამი მოდელი: RandomForest, XGBoost, Adaboost. ერთ-ერთი მთავარი პრობლემა, რასაც გადავეყარე, იყო ის, რომ GridSearch-ის გაშვების შემდეგ მონაცემთა დიდი რაოდენობის გამო Kaggle-ს მეხსიერება ივსებოდა და ნოუთბუქი იქრაშებოდა, ამიტომ ყველა ექსპერიმენტი ინდივიდუალურადაა გაშვებული.
მოდელის ტრენინგამდე ხდება მონაცემების Downsampling და 3.5%იანი თარგეთის წილის 30%მდე გაზრდა. თავდაპირველად სამივე მოდელს Downsampling-ის გარეშე ვაკეთებდი, რის გამოც მოდელებს დაბალი recall ჰქონდათ.

# *MLFlow tracking*
წრფივი რეგრესიის ექსპერიმენტი:
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_c1e8dd3f70c64bf68caf87b5383b6d1b

ხის მოდელის ექსპერიმენტი:
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_7702289a9828469bad55790437f20e5b

ხის მოდელის დამატებითი ექსპერიმენტები:
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_4928c2059ea041868a3e9104b5f566bc
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_4a2e3e471e764d298aa2d56b7ed4bbaa
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_1438ce557f1c48b386dde2fa394eb6a4
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_f789c6387cc24dbb831585c20df25d60
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_6413332904844893a426662f86283654
https://dagshub.com/dimna21/ML_Assignment1/experiments#/experiment/m_42c1cc9a778f4f21b76dbf4abccb660a
(ეს ექსპერიმენტები ჩავატარე, რადგან 0.856/0.81 R^2 შედეგები მაინც შეიძლება პატარა overfit-ად ჩაითვალოს. ამ ექსპერიმენტებში ხელით ვცვლიდი ხის სიღრმისა და ფოთლის გაყოფის რეინჯებს GridSearch-ისთვის და საუკეთესო გაუმჯობესება მინიმალური overfitის გაქრობის მხრივ იყო 0.84/0.81 მე-3 ექსპერიმენტში. ამ მოდელმა ტესტ სეტზე დაახლოებით იგივე შედეგი აჩვენა(0.18787 RMSLE), რაც საწყისმა მოდელმა, ეს კი იმის თქმის საფუძველს იძლევა, რომ მიკროსკოპული overfit პრობლემა არ ყოფილა და რეალურ გარემოში ორივე მოდელი თითქმის ანალოგიურ შედეგზე გადის)

MLFlow-თი ჩაწერილია training ნაწილში მოცემული dictionary-ს პარამეტრების სივრცე თითოეული მოდელისთვის და დალოგილია RMSE, MAE და R^2 მეტრიკები train/test სეტებისთვის.

საუკეთესო ხის მოდელმა Kaggle competition-ის ტესტ სეტზე აჩვენა 0.18204 RMSLE. იმის გათვალისწინებით, რომ leaderboardის თავში ხამები არიან, რომლებიც test set-ზე overfitting-ში ხარჯავენ თავისუფალ დროს და 0.00044 აქვთ ერორი, ამ შედარებით მარტივი არქიტექტურის მოდელებით მიღებული 0.18 ნორმალური შედეგია.

