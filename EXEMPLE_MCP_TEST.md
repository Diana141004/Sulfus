# 🧪 Exemple de Testare MCP — Copy-Paste în Chat

> Copiază oricare din prompturile de mai jos și lipește-le direct în chat-ul Cline.
> Serverul MCP `telecom-predictor` va fi apelat automat.

---

## ✅ Exemplu 1 — Client LOW RISK (ar trebui să NU riscă)

```
Analizează acest client telecom și spune-mi dacă va întârzia plata:
- Stat: OH
- Account length: 107 zile
- Area code: area_code_415
- Plan internațional: no
- Voicemail plan: yes
- Mesaje voicemail: 26
- Minute ziua: 161.6, apeluri ziua: 123
- Minute seara: 195.5, apeluri seara: 103
- Minute noaptea: 254.4, apeluri noaptea: 103
- Minute internaționale: 13.7, apeluri internaționale: 3
- Apeluri customer service: 1
```

**Rezultat așteptat:** NU riscă (prob ~13-15%)

---

## 🚨 Exemplu 2 — Client HIGH RISK (plan internațional + multe minute + multe apeluri CS)

```
Prezice dacă acest client va întârzia plata:
- Stat: CA
- Account length: 150 zile
- Area code: area_code_415
- Plan internațional: yes
- Voicemail plan: no
- Mesaje voicemail: 0
- Minute ziua: 265, apeluri ziua: 110
- Minute seara: 220, apeluri seara: 100
- Minute noaptea: 250, apeluri noaptea: 90
- Minute internaționale: 14.5, apeluri internaționale: 4
- Apeluri customer service: 4
```

**Rezultat așteptat:** RISCĂ (prob ~97%)

---

## 🚨 Exemplu 3 — Client VERY HIGH RISK (6 apeluri CS + consum masiv)

```
Analizează riscul de întârziere plată:
- State: NJ
- Account length: 200
- Area code: area_code_415
- International plan: yes
- Voice mail plan: no
- Number vmail messages: 0
- Total day minutes: 300, total day calls: 130
- Total eve minutes: 260, total eve calls: 110
- Total night minutes: 220, total night calls: 100
- Total intl minutes: 15, total intl calls: 2
- Number customer service calls: 6
```

**Rezultat așteptat:** RISCĂ (prob ~99%)

---

## ✅ Exemplu 4 — Client LOW RISK (voicemail activ, consum moderat, 0 apeluri CS)

```
Va întârzia plata clientul acesta?
- Stat: TX
- Zile de când e client: 45
- Cod zonă: area_code_510
- Plan internațional: no
- Plan voicemail: yes
- Mesaje voicemail: 30
- Minute/zi: 120, apeluri/zi: 80
- Minute seara: 150, apeluri seara: 70
- Minute noaptea: 180, apeluri noaptea: 90
- Minute internaționale: 8, apeluri internaționale: 5
- Apeluri customer service: 0
```

**Rezultat așteptat:** NU riscă (prob ~9%)

---

## 🚨 Exemplu 5 — Client din NY cu plan internațional și 5 apeluri CS

```
Folosește tool-ul predict_payment_delay pentru acest client:
State: NY, account_length: 80, area_code: area_code_408, international_plan: yes, 
voice_mail_plan: no, number_vmail_messages: 0, total_day_minutes: 230, 
total_day_calls: 120, total_eve_minutes: 180, total_eve_calls: 85, 
total_night_minutes: 180, total_night_calls: 95, total_intl_minutes: 12, 
total_intl_calls: 2, number_customer_service_calls: 5
```

**Rezultat așteptat:** RISCĂ (prob ~86%)

---

## 🔄 Exemplu 6 — Comparație threshold modes (același client, 2 moduri)

```
Analizează acest client cu AMBELE threshold modes (best_f1 și high_recall) și compară rezultatele:
- Stat: WA
- Account length: 120 zile
- Area code: area_code_415
- Plan internațional: yes
- Voicemail: no, mesaje: 0
- Minute ziua: 200, apeluri ziua: 100
- Minute seara: 170, apeluri seara: 90
- Minute noaptea: 190, apeluri noaptea: 85
- Minute internaționale: 11, apeluri internaționale: 3
- Apeluri customer service: 3
Folosește threshold_mode best_f1 și apoi high_recall.
```

**Rezultat așteptat:** 
- best_f1 (threshold 0.756): borderline — depinde de probabilitate
- high_recall (threshold 0.473): mai probabil RISCĂ (threshold mai permisiv)

---

## 🧪 Exemplu 7 — Prompt natural (fără structură)

```
Am un client din Florida, e client de 90 de zile, zona 510, are plan internațional dar nu are voicemail. Face cam 180 minute pe zi cu 95 apeluri, 160 minute seara cu 80 apeluri, 200 noaptea cu 100 apeluri, și 9 minute internaționale cu 4 apeluri. A sunat de 3 ori la customer service. Crezi că va întârzia plata?
```

---

## 📊 Exemplu 8 — Solicită și statisticile datasetului

```
Înainte de a analiza clientul, arată-mi statisticile datasetului de telecomunicații și apoi prezice dacă acest client va întârzia:
- State: IL, account_length: 130, area_code: area_code_408
- international_plan: no, voice_mail_plan: no, number_vmail_messages: 0
- total_day_minutes: 190, total_day_calls: 105
- total_eve_minutes: 200, total_eve_calls: 95
- total_night_minutes: 210, total_night_calls: 100
- total_intl_minutes: 10, total_intl_calls: 3
- number_customer_service_calls: 2
```

---

## 🎯 Exemplu 9 — Client "pe muchie" (borderline)

```
Vreau să testez un caz borderline. Analizează:
- Stat: MI
- Account length: 100
- Area code: area_code_415
- International plan: yes
- Voice mail plan: no
- Vmail messages: 0
- Day minutes: 200, day calls: 100
- Eve minutes: 180, eve calls: 90
- Night minutes: 200, night calls: 95
- Intl minutes: 10, intl calls: 3
- Customer service calls: 2

Ce zici, e risc sau nu? Explică-mi detaliat.
```

---

## 📋 Tabel Rezumat Factori de Risc

| Factor | Risc SCĂZUT | Risc RIDICAT |
|--------|-------------|--------------|
| Plan internațional | no | **yes** |
| Apeluri customer service | 0-1 | **3+** |
| Minute/zi | < 180 | **> 250** |
| Voicemail plan | yes | no |
| Mesaje voicemail | > 0 | 0 |

---

## 💡 Tips pentru Testare

1. **Factorii cei mai importanți** pentru predicție sunt:
   - `international_plan = yes` → crește riscul semnificativ
   - `number_customer_service_calls >= 4` → semnal puternic de risc
   - `total_day_minutes > 250` → consum excesiv = risc

2. **Threshold modes:**
   - `best_f1` (default) — echilibru optim, Precision 90.54%
   - `high_recall` — prinde mai mulți clienți cu risc, dar mai multe alarme false

3. **Cum verifici că MCP-ul funcționează:**
   - Cline ar trebui să apeleze automat tool-ul `predict_payment_delay`
   - Vei vedea în output parametrii trimiși și rezultatul returnat
