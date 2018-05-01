# -*;;;;;;;;;;;;;- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:10:04 2018

@author: frederico.c.silva
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:43:15 2018

@author: frederico.c.silva
"""
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.metrics import ConfusionMatrix


base_comentario_total = pd.read_table('train.tsv',usecols=[2,3])
base_comentario_predicao = pd.read_table('test.tsv',usecols=[0,2])

base_comentarios_treinamento,base_comentarios_teste=train_test_split(base_comentario_total,test_size=0.3, random_state=42)


base_comentarios_treinamento_lista=[]
for i in range(0,109241):
    base_comentarios_treinamento_lista.append([str(base_comentarios_treinamento.values[i,j]) for  j in range(0,2)])


base_comentarios_teste_lista=[]
for i in range(0,46817):
    base_comentarios_teste_lista.append([str(base_comentarios_teste.values[i,j]) for  j in range(0,2)])



#stopwordscompleto = nltk.corpus.stopwords.words('english')
#
#def removestopwords(texto):
#    frases=[]
#    for (Phrase,Sentiment) in texto:
#        Phrasesemstop = [p for p in Phrase.split() if p not in stopwordsçcompleto]
#        frases.append((Phrasesemstop,Sentiment))
#    return frases
#
#
#frases=removestopwords(base_comentarios_treinamento)


#####REMOVENDO OS STOPWORDS  e OS RADICAIS
######3REDUZIR AS DIMENSIONALIDADE DOS DADOS
stopwordscompleto = nltk.corpus.stopwords.words('english')


def RetirarStopWordsRadical(texto):
    radical=nltk.stem.SnowballStemmer('english')
    frasesSemRadical = []
    for (Phrase,Sentiment) in texto:
        FraseLimpa=[str(radical.stem(p)) for p in Phrase.split()  if p not in stopwordscompleto]
        frasesSemRadical.append((FraseLimpa,Sentiment))
    return frasesSemRadical






FrasesTreinamentoSemRadical =RetirarStopWordsRadical(base_comentarios_treinamento_lista)
FrasestesteSemRadical =RetirarStopWordsRadical(base_comentarios_teste_lista)



######PALAVRAS UNICAS
def buscapalavras(frases):
    todaspalavras=[]
    for (Phrase,Sentiment) in frases:
        todaspalavras.extend(Phrase)
    return todaspalavras





palavrasTreinamento = buscapalavras(FrasesTreinamentoSemRadical)
palavrasTeste = buscapalavras(FrasestesteSemRadical)

#####CONTABILIZANDO A FREQUENCIA DOS RADICAIS
def buscafrequencia(palavras):
    palavras=nltk.FreqDist(palavras)
    return palavras


frequenciaTreinamento = buscafrequencia(palavrasTreinamento)
frequenciaTeste = buscafrequencia(palavrasTeste)

#############PALAVRAS UNICAS
def buscapalavrasunicas(frequencia):
    freq=frequencia.keys()
    return freq

palavrasunicasTreinamento=buscapalavrasunicas(frequenciaTreinamento)
palavrasunicasTeste=buscapalavrasunicas(frequenciaTeste)

#print (palavraunicasPredicao)

###RETORNAR EXTRATOR DE PALAVRAS
def extratorpalavrasTreinamento(documento):
    doc = set(documento)
    caracteristicasTreinamento={}
    for palavras in palavrasunicasTreinamento:
        caracteristicasTreinamento['%s' % palavras]=(palavras in doc)
    return caracteristicasTreinamento


def extratorpalavrasTeste(documento):
    doc = set(documento)
    caracteristicasTeste={}
    for palavras in palavrasunicasTeste:
        caracteristicasTeste['%s' % palavras]=(palavras in doc)
    return caracteristicasTeste




#caracteristicasfrase = extratorpalavras()

basecompletaTreinamento = nltk.classify.apply_features(extratorpalavrasTreinamento,FrasesTreinamentoSemRadical)
basecompletaTeste = nltk.classify.apply_features(extratorpalavrasTeste,FrasestesteSemRadical)


#print(basecompleta)
#CONSTRUIR A TABELA DE PROBABILIDADES



#mostrando as classes

classificador = nltk.NaiveBayesClassifier.train(basecompletaTreinamento) ##constroi as tabelas de probabilidade
#print (classificador)

print(nltk.classify.accuracy(classificador,basecompletaTeste)) #acuracia


##############################################TESTE DE PREDIÇÃO########################

base_comentario_predicao_lista=[]
for i in range(0,66291):
    base_comentario_predicao_lista.append([str(base_comentario_predicao.values[i,j]) for  j in range(0,2)])




    def buscapalavrasPredicao(frases):
       todaspalavras=[]
       for (Phrase) in frases:
           todaspalavras.extend(FrasesPredicaoSemRadical)
       return todaspalavras







stopwordscompleto = nltk.corpus.stopwords.words('english')

for sentenca in base_comentario_predicao_lista:
 
    Phrase = sentenca[1]
#    print(Phrase)
    
    radical=nltk.stem.SnowballStemmer('english')
    frasesSemRadicalPredicao = []
    FrasesPredicaoSemRadical=[str(radical.stem(p)) for p in Phrase.split()  if p not in stopwordscompleto]
    #print(FrasesPredicaoSemRadical)
    
    palavrasTestePredicao=buscapalavrasPredicao(FrasesPredicaoSemRadical)
    #print(palavrasTestePredicao)
    
    
    FrequenciaPredicao = nltk.FreqDist(palavrasTestePredicao)
    #print(FrequenciaPredicao)
    
    
    palavraunicasPredicao=FrequenciaPredicao.keys()
    #print(palavraunicasPredicao)
    
    
    def extratorpalavrasPredicao(documento):
        doc = set(documento)
        caracteristicasPredicao={}
        for palavras in palavraunicasPredicao:
            caracteristicasPredicao['%s' % palavras]=(palavras in doc)
        return caracteristicasPredicao
    
    basecompletaPredicao =   extratorpalavrasPredicao(palavraunicasPredicao)
    
    print(sentenca[0]+','+classificador.classify(basecompletaPredicao))
    resultado=sentenca[0]+','+classificador.classify(basecompletaPredicao)
#    print(classificador.classify(basecompletaPredicao[0]))
    
    arquivo=open('resultado.txt','a')
    arquivo.write(resultado+'\n')
    arquivo.close()
###############################FIMMMM   DA       PREDICAO##########################################

#RecomendacaoPreditiva = nltk.ClassifierBasedTagger(train=basecompletaTreinamento,classifier_builder=classificador)

#print (classificador.classify(basecompletaTeste))
    
 

##############SCRIPT DE ERROS############MOSTRAR NA APRESENTAÇÃO
###EXISTE A NECESSIDADE DE UMA LINGUISTA PARA AJUDAR NESSA CLASSIFICAÇÃO
##AINDA TEM AS FRASES PALAVRAS QUE NÃO SIGNIFICAM NADA
erro=[]
for (Phrase,Sentiment) in basecompletaTeste:
    resultado=classificador.classify(Phrase)
    if resultado != Sentiment:
        erro.append((Sentiment,resultado,Phrase))
    
    
for (Sentiment,resultado,Phrase) in erro:
    print (Sentiment,resultado,Phrase)    

########################################

###################MONTANDO A MATRIZ DE CONFUSAO
print(nltk.classify.accuracy(classificador,basecompletaTeste)) #acuracia

esperado=[]
previsto=[]
for (Phrases,Sentiment) in basecompletaTeste:
    resultado=classificador.classify(Phrases)
    previsto.append(resultado)
    esperado.append(Sentiment)


#print(pre//////////////////visto)
#print(esperado)

matriz = ConfusionMatrix(esperado,previsto)
print(matriz)


#######estatisticas das classes



#########################################################

#PERGUNTAS A SEREM RESPONDIDAS
#1-avaliar o cenário - Precisao (medico -100 %) // precisao de emocao (no meio academico a partir de 60 % já eum bom valor)
#2-numero de classes - Acuracia foi de 53% e teriamos 20 % se fosse uma busca aleatória (o algoritmos passou neste teste)
#3 -ZeroRules - usar este algoritmo para medir outra acuracia Outra abordagem O % de sentimento que tem maior frquencia será o % de chance de uma nova avalição ter este mesmo sentimento

#analisar especialistas da área linguistica (se a palavra realmente tem relação com o sentimento)

#######################################################################

##mostrando relacao entre as palavras...importante mostrar isso############
#print (classificador.show_most_informative_features(100))

#teste='sad decline'
#testestemming =[]
#radical =nltk.stem.SnowballStemmer('english')
#for (palavras) in teste.split():
#    comstem=[p for p in palavras.split()]
#    testestemming.append(str(radical.stem(comstem[0])))
#
#novo=extratorpalavrastreinamento(testestemming)
#
#print(classificador.classify(novo))
#distribuicao=classificador.prob_classify(novo)    
#
#
##FUNCAO IMPORTANTE (pq mostra para cada caso a probabilidade da classe de acordo com o algoritmo)
#for classe in distribuicao.samples():
#    print("%s: %f" % (classe, distribuicao.prob(classe)))
    



