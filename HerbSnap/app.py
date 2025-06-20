from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
from io import BytesIO
import time

app = Flask(__name__)

# load trained model
model = load_model('./Model_Mobilenet.h5')

# define the target size for your model
target_size = (224, 224)


# function to preprocess the image
def preprocess_image(file):
    # use BytesIO to convert file into bytes
    img = image.load_img(BytesIO(file.read()), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/code')
def code():
    return render_template('code.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file'not in request.files:
        return render_template('home.html', prediction='no file part')
    file = request.files['file']

    if file.filename == '':
        return render_template('home.html', prediction='No selected file')

    time.sleep(2)

    img_array = preprocess_image(file)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    leavesArray = ["Aloevera",
                   "Amla",
                   "Amruthaballi",
                   "Arali",
                   "Astma_weed",
                   "Badipala",
                   "Balloon_Vine",
                   "Bamboo",
                   "Beans",
                   "Betel",
                   "Bhrami",
                   "Bringaraja",
                   "Caricature",
                   "Castor",
                   "Catharanthus",
                   "Chakte",
                   "Chilly",
                   "Citron lime (herelikai)",
                   "Coffee",
                   "Commonrue(naagdalli)",
                   "Coriender",
                   "Curry",
                   "Doddpathre",
                   "Drumstick",
                   "Ekka",
                   "Eucalyptus",
                   "Ganigale",
                   "Ganike",
                   "Gasagase",
                   "Ginger",
                   "Globe Amarnath",
                   "Guava",
                   "Henna",
                   "Hibiscus",
                   "Honge",
                   "Insulin",
                   "Jackfruit",
                   "Jasmine",
                   "Kambajala",
                   "Kasambruga",
                   "Kohlrabi",
                   "Lantana",
                   "Lemon",
                   "Lemongrass",
                   "Malabar_Nut",
                   "Malabar_Spinach",
                   "Mango",
                   "Marigold",
                   "Mint",
                   "Neem",
                   "Nelavembu",
                   "Nerale",
                   "Nooni",
                   "Onion",
                   "Padri",
                   "Palak(Spinach)",
                   "Papaya",
                   "Parijatha",
                   "Pea",
                   "Pepper",
                   "Pomoegranate",
                   "Pumpkin",
                   "Raddish",
                   "Rose",
                   "Sampige",
                   "Sapota",
                   "Seethaashoka",
                   "Seethapala",
                   "Spinach1",
                   "Tamarind",
                   "Taro",
                   "Tecoma",
                   "Thumbe",
                   "Tomato",
                   "Tulsi",
                   "Turmeric",
                   "ashoka",
                   "camphor",
                   "kamakasturi",
                   "kepala"]
                
    #usesArray = ["Aloe Vera: Also known as Aloe barbadensis, Aloe Vera is a succulent plant species of the genus Aloe. It is widely distributed and considered an invasive species in many world regions. An evergreen perennial, it originates from the Arabian Peninsula, but also grows wild in tropical, semi-tropical, and arid climates around the world. Medicinal Uses: Aloe Vera has been the subject of much scientific study over the last few years, regarding several claimed therapeutic properties. The leaves of Aloe Vera contain significant amounts of the polysaccharide gel acemannan, which can be used for topical purposes.Here are some of its uses: Skin Care: Aloe Vera gel has many medicinal properties and is often used in creams and lotions.It is renowned for treating skin injuries but also has several other uses that could potentially benefit health.Oral Health: A study published in General Dentistry reported that Aloe Vera in tooth gels is as effective as toothpaste in fighting cavities.Digestive Health: Germany's regulatory agency for herbs - Commission E - approved the use of Aloe Vera for the treatment of constipation.",
    
    # "Amruthaballi, or Tinospora Cordifolia, is a powerhouse of antioxidants which fight free-radicals, keep your cells healthy and get rid of diseases. It helps remove toxins, purifies blood, fights bacteria that cause diseases and also combats liver diseases and urinary tract infections. It is used by experts in treating heart related conditions, and is also found useful in treating infertility.In Ayurveda, it is considered a divine herb with its therapeutic benefits and is a part of the Rasayana group of herbs, which are used for rejuvenation, immune boosting, and in treatment of stress-induced disorders. It is also used in traditional medicine practices like Unani and Homeopathy for treating various diseases such as jaundice, diabetes, and skin diseases.It is also known to be beneficial for diabetic patients as it can lower blood sugar levels. It can also help in reducing respiratory problems like frequent cough, cold, tonsils.In addition to all these benefits, Amruthaballi is also used in skincare as it helps in reducing skin related diseases and inflammation. It is also known to enhance digestion and treat bowel related issues.It is truly a versatile herb and a gift of nature to mankind."
    
    
    # "Arali, also known as Nerium indicum, is a plant that has been used in Ayurvedic medicine for its various health benefits. It is also commonly known as Indian Oleander, Dog-bane, South-sea rose, and Adalpha.Medicinally, Arali has been found to have anti-inflammatory properties that can help ease the burning sensation and pain caused by inflammations in the body. It also has adaptogenic properties that help regulate blood pressure. Additionally, it addresses motor-control issues, alleviating symptoms of conditions such as vertigo, epilepsy, and seizures.Being an antiseptic, Arali is also beneficial for addressing skin-related conditions. It can penetrate the blood-brain barrier, providing oxygen to brain cells, improving cognitive function and memory.However, it's important to note that while Arali has many potential health benefits, it should be used under the guidance of a healthcare professional due to its toxic nature when consumed internally.",

    # "Badipala is a common name for a plant and it might refer to different species in different regions. Unfortunately, I couldn't find the exact scientific name for a plant referred to as 'Badipala'. It's possible that it's a regional or local name, and it might be helpful to'have more context or the English name of the plant to provide accurate information.However, there is a plant known as Bryophyllum pinnatum, also known as Patharchatta, which is rich in medicinal qualities and beneficial in curing the body of numerous ailments1. Over the years, Bryophyllum pinnatum has been used to treat various infections connected to renal and urinary issues. It has numerous health benefits for the body besides clearing the stomach of toxins and healing stones1.",
      
      
    # "Balloon Vine, also referred to as Mudakathan keerai, holds a place in traditional medicine for its anti-inflammatory properties. It is believed to alleviate joint pain and inflammation when consumed.",
    
    # "Bamboo, known for its versatile uses, extends beyond construction. Medicinally, it is believed to have anti-bacterial and anti-viral properties. Bamboo shoots, a nutritious food source, are consumed.",
    
    # "Beans, a staple vegetable, is rich in protein and fiber. Regular consumption promotes digestive health, regulates blood sugar levels, and supports heart health.",
    
    # "The scientific name of the Betel leaf is Piper betle. It belongs to the Piperaceae family. Betel leaf is a heart-shaped perennial creeper and is found in tropical, sub-tropical countries like India, Sri Lanka, Malaysia, Indonesia, Philippines, and East Africa.Here are some of the medicinal uses of Betel leaves:Anti-diabetic Agent: Dried betel leaf powder has the ability to reduce blood glucose levels in individuals with newly diagnosed type 2 diabetes mellitus5.Lowers High Cholesterol Levels: Betel leaf helps in lowering high levels of total cholesterol, triglycerides, low-density lipoprotein (LDL) cholesterol, and very low-density lipoprotein (VLDL) cholesterol5.Antioxidant: Betel leaf is a great source of antioxidant that fights oxidative stress by scavenging free radicals4.Anti-cancer: Betel leaf has strong anti-cancer and anti-mutagenic compounds in it and can lower the risk of developing cancer.Anti-microbial: Betel leaf has anti-microbial properties that can protect you from minor bacterial and fungal infections.Wound Healing: Betel leaf can help heal wounds, especially burn wounds.Treats Ear Infections, Headaches, Stomach Pain, and Insect Bites: Betel leaf treats ear infections, headaches, stomach pain, and even insect bites.",
     
    # "Bhrami, a revered medicinal herb, is known for its cognitive benefits. It is believed to enhance memory, reduce stress, and promote overall mental well-being in Ayurvedic medicine.",
     
    # "Bringaraja, also known as false daisy, is valued for its hair care properties. Used in traditional medicine, it promotes hair growth and maintains scalp health.",
      
    # "Caricature, a flowering plant, is known for its ornamental value. Its vibrant flowers are used in garlands and decorative arrangements, adding beauty to gardens and events.",
      
    # "Castor, with oil production as its main use, has medicinal benefits. Castor oil is employed for various skin conditions, hair care, and as a laxative in traditional medicine.",
      
    # "Catharanthus, commonly known as periwinkle, holds medicinal significance. It is used in chemotherapy and is recognized for its anti-cancer properties in traditional medicine.",
      
    # "Chakte, a tree with hardwood, has traditional uses in woodworking. The wood is employed for crafting furniture and various tools due to its durability and strength.",
      
      
    # "Chilly, a popular spice, adds flavor and heat to dishes. Besides culinary uses, it is believed to have medicinal properties, including pain relief and improved metabolism.",
       
       
    # "Citron lime (herelikai), prized for its unique flavor, is used in culinary applications. Rich in Vitamin C, it contributes to immune health and is employed in traditional medicine.",
       
    # "Coffee, a beloved beverage, is cultivated for its beans. Besides being a stimulant, coffee is believed to have antioxidant properties and potential health benefits when consumed in moderation.",
        
    # "Common rue(naagdalli), also known as rue or garden rue, has historical uses. It is believed to have medicinal properties and is employed in traditional practices for various purposes.",
        
    # "Coriander, a versatile herb, is widely used in culinary dishes. It adds flavor to meals and is believed to have digestive benefits and anti-inflammatory properties in traditional medicine.",
         
    # "Curry, a mix of aromatic spices, is a staple in various cuisines. Besides culinary uses, it is believed to have medicinal properties, including anti-inflammatory and antioxidant effects.",
         
    # "Doddpathre, known as Indian spinach, is a leafy green vegetable. Rich in nutrients, it is used in culinary dishes and is believed to contribute to overall health when consumed regularly.",
         
    # "Drumstick, recognized for its long, slender pods, is used in culinary dishes. It is valued for its nutritional content and is believed to have anti-inflammatory properties.",
          
    # "Ekka, a traditional medicinal plant, is used for its healing properties. It is believed to have anti-inflammatory effects and is employed in traditional medicine for various ailments.", "Eucalyptus, known for its fragrant leaves, has medicinal uses. Eucalyptus oil is recognized for its respiratory benefits and is employed in aromatherapy and traditional medicine.", "Ganigale, a type of aromatic grass, is used in traditional medicine. It is believed to have anti-inflammatory properties and is employed for its medicinal benefits.",
           
    # "Ganike, also known as Indian borage, is a medicinal herb. It is used for its anti-inflammatory and antibacterial properties in traditional medicine.",
           
    # "Gasagase, or poppy seeds, are culinary seeds with health benefits. They are believed to have nutritional value and are used in cooking for flavor and texture.",
           
    # "Ginger, a versatile spice, is used in culinary dishes and traditional medicine. It adds flavor, aids digestion, and is believed to have anti-inflammatory and antioxidant properties.",
           
    # "Globe Amarnath, known for its vibrant flowers, is used ornamentally. The flowers are used in garlands and decorative arrangements, adding color to gardens and events.",
           
    # "Guava, a tropical fruit, is valued for its sweet taste and nutritional content. It is rich in Vitamin C, fiber, and antioxidants, contributing to overall health when consumed.",
           
    # "The scientific name of the Henna plant is Lawsonia inermis1234. It belongs to the Lythraceae family3. Henna is a perennial shrub, which is commercially grown for leaf production.Here are some of the medicinal uses of Henna:Anti-hemorrhagic: Henna is used as an anti-hemorrhagic agent6.Anti-neoplastic: Henna has intestinal anti-neoplastic properties6.Cardio-inhibitory: Henna has cardio-inhibitory effects6.Hypotensive: Henna is used as a hypotensive6.Sedative: Henna is used as a sedative6.Anti-microbial: Henna extracts exhibit antibacterial, antifungal, and ultraviolet light screening activity6.Pain Relief: The oil of its flower relieves muscular pains6.Regulates Menstruation: Its seeds are used to regulate menstruation6.Treats Skin Conditions: Henna is sometimes applied directly to the affected area for dandruff, eczema, scabies, fungal infections, and wounds7.Prevents Diseases: Henna leaves grinded with date seeds and mustard seed and applying this paste on the palm may prevent pox and other skin diseases6.Hair Care: Henna leaves can rejuvenate the hair color. Moreover, it also prevents related hair problems like - dandruff, hair fall.",
           
    # "Hibiscus, with its vibrant flowers, is used ornamentally and medicinally. Hibiscus tea is believed to have health benefits, including antioxidant properties and potential blood pressure regulation.",
           
    # "Honge, also known as Indian Beech, has versatile uses. The wood is used in construction, and the plant is valued for its medicinal properties in traditional medicine.",
           
    # "Insulin plant, recognized for its insulin-like properties, is used in traditional medicine. It is believed to help regulate blood sugar levels and is employed for its potential anti-diabetic effects.",
           
    # "Jackfruit, a large tropical fruit, is used for its edible seeds and flesh. It is a versatile ingredient in various culinary dishes and is valued for its nutritional content.",
           
    # "Jasmine, known for its fragrant flowers, is used ornamentally. Jasmine flowers are used in perfumery, garlands, and decorative arrangements, adding a sweet scent to gardens and events.",
           
    # "Kambajala, a traditional medicinal plant, is used for its healing properties. It is believed to have anti-inflammatory effects and is employed in traditional medicine for various ailments.",
           
    # "Kasambruga, a traditional medicinal herb, is used for its healing properties. It is believed to have anti-inflammatory effects and is employed in traditional medicine for various ailments.",
           
    # "Kohlrabi, a vegetable with a unique appearance, is used in culinary dishes. It is valued for its crisp texture and mild", "Lantana, known for its colorful flowers, is used ornamentally. Lantana flowers are used in gardens and landscaping, adding vibrant colors to outdoor spaces.",
           
    # "Lemon, a citrus fruit, is valued for its refreshing taste and nutritional content. It is rich in Vitamin C, antioxidants, and is commonly used in culinary dishes and beverages.",
           
    # "Lemongrass, known for its citrus flavor, is used in culinary dishes and herbal teas. It is valued for its refreshing taste and is believed to have potential health benefits.",
           
    # "Malabar Nut, a medicinal herb, is used for its respiratory benefits. It is believed to help alleviate respiratory issues and is employed in traditional medicine for its healing properties.",
           
    # "Malabar Spinach, a leafy green vegetable, is used in culinary dishes. It is rich in nutrients, including iron, and is valued for its unique taste and texture.",
           
    # "Mango, a popular tropical fruit, is enjoyed for its sweet taste and versatility. It is used in culinary dishes, beverages, and is valued for its nutritional content.",
           
    # "Marigold, known for its vibrant flowers, is used ornamentally. Marigold flowers are used in garlands, decorative arrangements, and as offerings, adding color to gardens and events.",
           
    # "Mint, a fragrant herb, is used in culinary dishes and beverages. It adds a refreshing flavor and is believed to have digestive and respiratory benefits.",
           
    # "Neem, a versatile medicinal tree, is used for its various health benefits. Neem leaves, oil, and bark are employed in traditional medicine for their antibacterial and antifungal properties.",
           
    # "Nelavembu, a traditional medicinal plant, is used for its healing properties. It is believed to have immune-boosting effects and is employed in traditional medicine for various ailments.",
           
    # "Nerale, also known as Indian gooseberry, is used in culinary dishes. It is valued for its tangy flavor and is used in pickles, chutneys, and beverages.",
           
    # "Nooni, a traditional medicinal plant, is used for its healing properties. It is believed to have anti-inflammatory effects and is employed in traditional medicine for various ailments.",
           
    # "Onion, a staple vegetable, is used in culinary dishes worldwide. It adds flavor to meals and is valued for its versatility in various savory recipes.",
           
    # "Padri, known for its aromatic leaves, is used in traditional medicine. The leaves are believed to have medicinal properties and are employed for their potential health benefits.",
           
    # "Palak(Spinach), a nutrient-rich leafy green, is used in culinary dishes. It is valued for its iron content and is a popular ingredient in salads, curries, and smoothies.",
           
    # "Papaya, a tropical fruit, is enjoyed for its sweet taste and nutritional content. It is rich in Vitamin C, antioxidants, and is commonly used in culinary dishes and beverages.",
           
    # "Parijatha, known for its fragrant flowers, is used ornamentally. Parijatha flowers are used in garlands, decorative arrangements, and as offerings, adding elegance to gardens and events.",
           
    # "Pea, a versatile legume, is used in culinary dishes. Peas are a good source of protein, fiber, and various vitamins, contributing to a well-balanced diet.",
           
    # "Pepper, a popular spice, adds heat and flavor to dishes. Besides culinary uses, black pepper is believed to have digestive benefits and potential anti-inflammatory effects in traditional medicine.",
           
    # "Pomoegranate, a nutrient-rich fruit, is enjoyed for its sweet and tart taste. It is rich in antioxidants and is valued for its potential health benefits when consumed regularly.",
           
    # "Pumpkin, a versatile vegetable, is used in culinary dishes worldwide. It is valued for its sweet and nutty flavor, as well as its nutritional content.",
           
    # "Raddish, a root vegetable, is used in culinary dishes. It adds a crisp texture and peppery flavor to salads, sandwiches, and other recipe",
           
    # "Rose, known for its fragrant flowers, is used ornamentally. Rose flowers are used in perfumery, garlands, and decorative arrangements, adding a delightful scent to gardens and events.",
           
    # "Sampige, known for its fragrant flowers, is used ornamentally. Sampige flowers are used in garlands, decorative arrangements, and as offerings, adding beauty to gardens and events.",
           
    # "Sapota, a tropical fruit, is enjoyed for its sweet and grainy texture. It is commonly used in culinary dishes, desserts, and beverages.",
           
    # "Seethaashoka, known for its vibrant flowers, is used ornamentally. Seethaashoka flowers are used in garlands, decorative arrangements, and as offerings, adding color to gardens and events.",


    # "Seethapala, a tropical fruit, is enjoyed for its sweet and creamy texture. It is commonly used in culinary dishes, desserts, and beverage",
                 
    # "Spinach, a nutrient-rich leafy green, is used in culinary dishes. It is valued for its iron content and is a popular ingredient in salads, curries, and smoothies.", "Tamarind, a tangy fruit, is used in culinary dishes worldwide. Tamarind pulp adds a distinct flavor to meals and is commonly used in sauces, chutneys, and beverages.",
                
    # "Taro, a starchy root vegetable, is used in culinary dishes. It has a creamy texture and is commonly used in stews, soups, and various Asian cuisines.", "Tecoma, a flowering plant, is used ornamentally. Tecoma flowers are used in gardens and landscaping, adding vibrant colors to outdoor spaces.", "Thumbe, known for its medicinal properties, is used in traditional medicine. It is believed to have anti-inflammatory effects and is employed for various health benefits.",
                 
    # "Tomato, a versatile fruit, is used in culinary dishes worldwide. Tomatoes are rich in antioxidants, vitamins, and are a staple ingredient in salads, sauces, and soups.",
                  
    # "Tulsi, known as holy basil, is a revered medicinal herb. It is valued for its potential health benefits, including stress relief, immune support, and respiratory benefits.", "Turmeric, a golden spice, is used in culinary dishes and traditional medicine. It adds color and flavor to meals and is believed to have anti-inflammatory and antioxidant properties.",
                 
    # "ashoka, known for its vibrant flowers, is used ornamentally. Ashoka flowers are used in garlands, decorative arrangements, and as offerings, adding beauty to gardens and events.", "Camphor used in traditional medicine and aromatherapy for respiratory relief. Aromatic properties make it popular in therapeutic applications.", "Kamakasturi known for its pleasant fragrance, used in perfumery. Holds cultural significance, often incorporated into traditional ceremonies or practices.",
                 
    # "Kepala with potential health benefits, used in traditional medicine. Culinary uses for flavor and ornamental value in gardens."]


    usesArray = [
        "Aloe vera: Soothes skin irritations, promotes wound healing, and aids digestion.",
        "Amla: Rich in vitamin C, boosts immunity, and promotes hair health.",
        "Amruthaballi (Tinospora cordifolia): Enhances immunity and supports liver health.",
        "Arali (Nerium oleander): Ornamental plant; note that all parts are toxic if ingested.",
        "Asthma weed (Euphorbia hirta): Traditionally used to treat respiratory conditions like asthma.",
        "Badipala (Justicia adhatoda): Known for its expectorant properties, helps in respiratory ailments.",
        "Balloon Vine (Cardiospermum halicacabum): Used in traditional medicine for joint pain and skin disorders.",
        "Bamboo: Versatile plant used in construction, crafts, and as a food source (bamboo shoots).",
        "Beans: High in protein and fiber; beneficial for heart health.",
        "Betel (Piper betle): Leaves used in traditional practices; has digestive and stimulant properties.",
        "Brahmi (Bacopa monnieri): Enhances cognitive functions and memory.",
        "Bhringraj (Eclipta prostrata): Promotes hair growth and liver health.",
        "Caricature plant (Graptophyllum pictum): Used in traditional medicine for treating hemorrhoids and earaches.",
        "Castor (Ricinus communis): Seeds produce castor oil, used as a laxative and in skin care.",
        "Catharanthus (Catharanthus roseus): Source of alkaloids used in cancer treatment.",
        "Chakte (Caesalpinia sappan): Wood yields a red dye; used in traditional medicine.",
        "Chili (Capsicum spp.): Contains capsaicin; boosts metabolism and relieves pain.",
        "Citron lime (Herelikai): Rich in vitamin C; used in culinary and medicinal applications.",
        "Coffee (Coffea spp.): Stimulant; improves alertness and contains antioxidants.",
        "Common rue (Ruta graveolens): Used in traditional medicine for digestive issues; caution advised due to potential toxicity.",
        "Coriander (Coriandrum sativum): Leaves and seeds used in cooking; aids digestion.",
        "Curry leaf (Murraya koenigii): Enhances flavor; supports liver function and digestion.",
        "Doddapatre (Coleus amboinicus): Used for respiratory issues and as a culinary herb.",
        "Drumstick (Moringa oleifera): Nutrient-rich; supports overall health.",
        "Ekka (Calotropis gigantea): Used in traditional medicine; caution due to toxicity.",
        "Eucalyptus: Leaves produce oil with antiseptic properties; used in respiratory treatments.",
        "Ganigale (Clerodendrum serratum): Traditionally used for respiratory ailments.",
        "Ganike (Celosia argentea): Leaves consumed as leafy vegetable; used in traditional medicine.",
        "Gasagase (Papaver somniferum): Seeds used in cooking; have calming effects.",
        "Ginger (Zingiber officinale): Aids digestion; anti-inflammatory properties.",
        "Globe Amaranth (Gomphrena globosa): Ornamental; used in herbal teas.",
        "Guava (Psidium guajava): Rich in vitamin C; supports immune health.",
        "Henna (Lawsonia inermis): Leaves used as natural dye; has cooling properties.",
        "Hibiscus (Hibiscus rosa-sinensis): Flowers used in teas; supports heart health.",
        "Honge (Pongamia pinnata): Seeds yield oil used as biofuel and in traditional medicine.",
        "Insulin plant (Costus igneus): Believed to help regulate blood sugar levels.",
        "Jackfruit (Artocarpus heterophyllus): Nutrient-rich fruit; seeds also edible.",
        "Jasmine (Jasminum spp.): Flowers used in perfumes; has calming effects.",
        "Kambajala (Alternanthera sessilis): Leaves consumed as leafy vegetable; used in traditional medicine.",
        "Kasambruga (Michelia champaca): Flowers used in perfumes; ornamental tree.",
        "Kohlrabi (Brassica oleracea): Edible stem; rich in fiber and vitamin C.",
        "Lantana (Lantana camara): Ornamental; caution as plant parts are toxic if ingested.",
        "Lemon (Citrus limon): Rich in vitamin C; aids digestion.",
        "Lemongrass (Cymbopogon citratus): Used in teas; has calming effects.",
        "Malabar nut (Justicia adhatoda): Used in traditional medicine for respiratory issues.",
        "Malabar spinach (Basella alba): Leafy vegetable rich in vitamins A and C.",
        "Mango (Mangifera indica): Fruit rich in vitamins; leaves used in traditional medicine.",
        "Marigold (Tagetes spp.): Flowers used in rituals; has pest-repellent properties.",
        "Mint (Mentha spp.): Aids digestion; used in culinary applications.",
        "Neem (Azadirachta indica): Antibacterial properties; used in skin care.",
        "Nelavembu (Andrographis paniculata): Known for immune-boosting properties.",
        "Nerale (Syzygium cumini): Fruit supports blood sugar regulation.",
        "Nooni (Morinda citrifolia): Fruit used in traditional medicine; supports immune health.",
        "Onion (Allium cepa): Contains antioxidants; supports heart health.",
        "Padri (Clerodendrum infortunatum): Used in traditional medicine for various ailments.",
        "Palak (Spinach) (Spinacia oleracea): Rich in iron and vitamins; supports overall health.",
        "Papaya (Carica papaya): Aids digestion; rich in vitamin C.",
        "Parijatha (Nyctanthes arbor-tristis): Flowers used in traditional medicine; ornamental.",
        "Pea (Pisum sativum): High in protein and fiber; supports heart health.",
        "Pepper (Piper nigrum): Enhances flavor; aids digestion.",
        "Pomegranate (Punica granatum): Rich in antioxidants; supports heart health.",
        "Pumpkin (Cucurbita spp.): Rich in vitamins; seeds are nutritious.",
        "Radish (Raphanus sativus): Supports digestion; rich in vitamin C.",
        "Rose (Rosa spp.): Petals used in cosmetics; has calming effects.",
        "Sampige (Magnolia champaca): Fragrant flowers used in perfumes.",
        "Sapota (Manilkara zapota): Sweet fruit; rich in vitamins.",
        "Seethaashoka (Saraca asoca): Bark used in traditional medicine for women's health.",
        "Seethapala (Annona squamosa): Sweet fruit; rich in nutrients.",
        "Spinach1 (Spinacia oleracea): Duplicate entry; see Palak.",
        "Tamarind (Tamarindus indica): Pulp used in cooking; aids digestion.",
        "Taro (Colocasia esculenta): Edible tuber; rich in carbohydrates.",
        "Tecoma (Tecoma stans): Ornamental plant; flowers attract pollinators.",
        "Thumbe (Leucas aspera): Used in traditional medicine for respiratory issues.",
        "Tomato (Solanum lycopersicum): Rich in lycopene; supports heart health.",
        "Tulsi (Ocimum tenuiflorum): Sacred plant; boosts immunity.",
        "Turmeric (Curcuma longa): Anti-inflammatory properties; used in cooking.",
        "Ashoka (Saraca asoca): Duplicate entry; see Seethaashoka.",
        "Camphor (Cinnamomum camphora): Used in rituals; has medicinal properties.",
        "Kamakasturi (Ocimum kilimandscharicum): Aromatic herb; used in traditional medicine.",
        "Kepala (Musa spp.): Refers to banana; fruit is nutritious and energy-rich."
    ]

    answer = leavesArray[predicted_class]
    use = usesArray[predicted_class]
    l = len(usesArray)
    print(l)

    return render_template('prediction.html', prediction=f'Predicted plant : {answer} Leaf', predictionUse=f'{use}')


#if __name__ == '__main__':
    #app.run(host='192.168.31.67', port='5500', debug=True)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

