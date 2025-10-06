[My Linkedin](https://www.linkedin.com/in/jos%C3%A9-eduardo-souza-leite/)
- pip install -r requirements.txt

# Exercise 1 - Module 36

<img width="391" height="126" alt="image" src="https://github.com/user-attachments/assets/ff22342c-3737-49a5-a7e1-c7b1da24c61a" />

## Dataset

This database comes from the **AMABiliDados** project and contains data from São Paulo tax receipts registered for automatic donation to AMA, the Association of Friends of Autistic People.

*Nota Fiscal Paulista* is a consumer incentive program from the São Paulo state government, which returns a portion of the ICMS tax whenever you request that your CPF be included on the receipt. 
Under this program, taxpayers can direct their credits to an NGO, and when they do so, whenever the registered consumer makes a purchase, the credits from receipts issued by the same unidentified establishment (receipts without a CPF) are "dragged" to the NGO in the form of a donation.

- [Click here](https://www.ama.org.br/site/) if you want to learn more about AMA.
- [Click here](https://doacao1.ama.org.br/sitenfp) if you want to learn how you can become an NFP donor.

This database contains data on tax invoices that have allocated their credits to AMA. Its fields are described below:

|Field|Description|
|:-|:-|
|CNPJ emit.| CNPJ of the invoice issuer|
|Emitente| Business name of the invoice issuer|
|No.| Invoice Number|
|Data Emissão| Invoice issuance data|
|Valor NF| Invoice Amount|
|Data Registro| Registration data in the NFP system|
|Créditos| Credit Amount (donation)|
|Situação do Crédito| Whether the credit has already been paid, is being processed, etc.|
|Ano| Year of invoice issuance|
|Semestre| Semester of invoice issuance|
|Retorno| Credit amount divided by the invoice amount|
|flag_credito| Indicates whether the invoice has positive credit|
|categoria| Note categorization |

---

## Objective
Some invoices don't provide returns, which can occur for a variety of reasons, including the presence of non-incentivized products such as cigarettes and alcoholic beverages. Our goal is to predict which types of invoices are more or less likely to provide credits.

**Note**: This is a real database, with consistent characteristics. There may be variability that is difficult to explain, partly due to the frequency of invoices that varies over time (for various reasons), and partly because important information, protected by the LGPD, is not included in the database, which may interfere with the credit generated, resulting in variability that is difficult to explain.

---

## Analysis of the probability of return

```python
# variavel para criar um ordem descrente no gráfico: maior valor para o menor.
ordem = (
    new_df.groupby('categoria')['Retorno'].sum()  # Soma os retornor por categoria                               
    .sort_values(ascending=False)                 # Coloca do maior para o menor
    .index                                        
)

# Gráfico barplot
sns.barplot(
    data=new_df,
    x='categoria',
    y='Retorno',
    order=ordem,
    hue='categoria',
    estimator=sum,
    errorbar=None,
    palette='Set1')

plt.title("Retorno por Categoria")
plt.xlabel("Categorias")
plt.ylabel("Retorno")
plt.xticks(rotation=45)
plt.tight_layout()   
plt.grid(alpha=.2)
plt.show()


# Informações descritivas do gráfico
print(new_df.groupby('categoria')['Retorno'].sum().sort_values(ascending=False).round(2))
```

<img width="622" height="649" alt="image" src="https://github.com/user-attachments/assets/476ae341-80f9-4bc6-be7f-dd1934b1a569" />

## Graph Analysis

At first glance, we see that the first three variables account for more than **70%** of the total estimated return, which may indicate:

- A greater number of invoices issued;
- Higher average values;
- Higher individual return probabilities

Popular sectors, such as *Food* and *Restaurants*, have good volume, but lower returns than *Construction*, for example.

**Auto Posto**, **Varejo**, and **Vestuário** are segments with a very low probability of return. It is worth studying the possibility of grouping these categories.

I emphasize that the **Não Definido** category has the highest value in the sample. This is informative, but requires cleaning, as there may be noise and a mix of different segments.

---

## WOE

```python
# Fazendo uma cópia de segurança do DataFrame original
df_crosstab = new_df.copy()

# Criando uma nova coluna com o filtro de Retorno positivo, ou seja, acima de zero.
df_crosstab['retorno_positivo'] = (df_crosstab['Retorno'] > 0).astype(int)

# Tabelea cruzada com as categorias e retorno positivo
tab = pd.crosstab(df_crosstab['categoria'],
                  df_crosstab['retorno_positivo'],
                  margins=True,
                  margins_name='Total')

# Variáveis que indicam se o evento é positivo ou negativo
afirmativo = tab.columns[1]
negativo = tab.columns[0]

# Cálculo do percentual do evento
tab['%_nao_evento'] = tab[negativo]/tab.loc['Total',negativo]
tab['%_evento'] = tab[afirmativo]/tab.loc['Total',afirmativo]

# Cálculo da Chance de Risco, ou seja, odds ratio do evento.
tab['odds_ratio'] = tab['%_evento']/tab['%_nao_evento']

# Cálculo da força e direção da relação da categoria com o evento, o WOE. Sendo este, o log de Odds Ratio.
tab['woe'] = np.log(tab['odds_ratio'])
```

<img width="583" height="371" alt="image" src="https://github.com/user-attachments/assets/54e846a9-fdb8-45b3-837c-6aaf8f374339" />

```python
sns.barplot(
    data=tab,
    x='categoria',
    y='woe',
    order=ordem,
    hue='categoria',
    estimator=sum,
    errorbar=None,
    palette='Set1')

plt.title("WOE por Categoria")
plt.xlabel("Categorias")
plt.ylabel("Weigh of Evidence")
plt.xticks(rotation=45)
plt.grid(alpha=.2)
plt.tight_layout()       
plt.show()
plt.show()
```

<img width="634" height="461" alt="image" src="https://github.com/user-attachments/assets/0dade31d-8554-4938-80ed-5abe83ecd444" />

This analysis shows the **average probability of return** (`Retorno`) for each `categoria`. Higher values indicate higher expected contribution to positive return (e.g., donations via NFP to AMA).

| Categoria       | Retorno Médio | Interpretação |
|:---------------|-------------:|:-------------|
| não definido    | 3741.91      | High value, likely contains unclassified or mixed records. Requires cleaning. |
| Construção      | 3422.49      | High return probability; sector contributes significantly to positive outcomes. |
| Mercado         | 3032.58      | Strong contribution, frequent transactions with engaged customers. |
| Alimentos       | 1258.80      | Medium return; stable and relevant segment. |
| Restaurantes    | 887.59       | Good relative performance; indicates engaged donors. |
| Farmácia        | 399.35       | Lower probability, still contributes; smaller average transaction value. |
| Vestuário       | 370.16       | Low probability; seasonal or less frequent purchases. |
| Varejo          | 154.12       | Very low probability; weak engagement or low valid note volume. |
| Auto posto      | 9.92         | Extremely low; almost no positive return observed. |

### Key Insights

1. **Concentration of return**  
   - `não definido`, `Construção`, and `Mercado` contribute over 70% of total return. Indicates strong concentration in a few categories.

2. **Popular sectors not always high return**  
   - `Alimentos` and `Restaurantes` have moderate average return. Frequency of purchases does not always imply high return.

3. **Data cleaning opportunity**  
   - `não definido` has abnormally high return. Investigate and redistribute records for better accuracy.

4. **Low-performing sectors**  
   - `Auto posto`, `Varejo`, and `Vestuário` show low return. Could be targeted for educational campaigns or deprioritized in models.

5. **Consistency with WOE/IV analysis**  
   - Categories with positive WOE (`Alimentos`, `Restaurantes`, `Mercado`) maintain reasonable returns.
   - Categories with negative WOE (`Varejo`, `Auto posto`) maintain low returns, confirming previous segmentation.

---

## Information Value

```python
IV_classificacao = {
    (0, 0.02): "Inútil",
    (0.02, 0.1): "Fraco",
    (0.1, 0.3): "Médio",
    (0.3, 0.5): "Forte",
    (0.5, np.inf): "Suspeito de tão alto"
}

def classificar_iv(valor):
    for (inicio, fim), classificacao in IV_classificacao.items():
        if inicio <= valor < fim:
            return classificacao
    return None

tab['IV'] = (tab['%_evento'] - tab['%_nao_evento'])*tab['woe']
tab['Siddiqi_classificador'] = tab['IV'].apply(classificar_iv)

iv = tab['IV'].sum().round(4)
print(f'O IV total da variável categoria é {iv}.')

tab
```

<img width="792" height="354" alt="image" src="https://github.com/user-attachments/assets/8f228301-71e7-4131-9dbe-9f8fcd16ae9e" />

Following **Siddiqi's rules** for IV interpretation:

| Faixa       | Poder preditivo |
|:-----------|:----------------|
| 0 a 0,02   | Inútil         |
| 0,02 a 0,1 | Fraco            |
| 0,1 a 0,3  | Médio          |
| 0,3 a 0,5  | Forte          |
| 0,5 ou mais| Suspeito de tão alto|

---

- The <font color='red'>**Varejo**</font> category has the **highest individual IV** (*0.107*).  
  Despite being associated with a **lower probability of return** (*negative WOE*), it is the category that **contributes most to separating events vs non-events**, so it deserves attention in modeling.

- The <font color='blue'>**Não Definido**</font> category also deserves careful analysis.  
  Its significant volume of data may introduce noise, and clarifying or redistributing it could **improve the IV**.

- Some categories are **useless individually**, such as `Farmácia`, `Auto Posto` and `Vestuário`.  
  It may be worth considering **grouping them with similar categories** to reduce sparsity.

- There is a clear **balance and direction** among categories:  
  - Opposite behavior to donors: *Varejo*, *Vestuário* (WOE negative)  
  - Concentration of engaged donors: *Restaurantes*, *Alimentos* (WOE positive)

---

**Conclusion**  
Even though some categories have low individual *IV*, **the variable as a whole is useful for separating events from non-events**.  
The **total IV** of the variable is *0.279*, indicating **medium predictive power**.
