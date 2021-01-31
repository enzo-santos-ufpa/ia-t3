# Análise comparativa entre árvores de decisão e redes neurais em um problema de classificação binária

por
Enzo G. Santos,
Lucas M. Ferreira


## Resumo

Esse artigo faz parte da avaliação final da disciplina de Inteligência
Artificial e tem como objetivo comparar dois algoritmos de classificação:
árvores de decisão e redes neurais utilizando *multi-layer perceptrons*. O
problema de classificação escolhido envolve definir se um indivíduo possui
diabetes em um estágio inicial por meio de atributos como idade e perda de
peso. Cada algoritmo foi aplicado sobre a base de dados e analisadas
métricas que incluem a acurácia e a matriz de confusão da classificação.
Por fim, as métricas são comparadas e escolhe-se o melhor algoritmo para
a tarefa.


## Implementação

Todo o código-fonte usado para gerar os resultados e figuras no artigo 
estão na pasta `code`.
Os cálculos de métricas são rodados dentro de
[notebooks Jupyter](http://jupyter.org/).
A base de dados utilizada nesse estudo está disponibilizada em `data`.


## Obtendo o código

É possível baixar uma cópia de todos os arquivos nesse repositório clonando
o seu repositório [git](https://git-scm.com/):

    git clone https://github.com/enzo-santos/dt-nn-comparison.git
    

## Reproduzindo os resultados

Para explorar os resultados do código, é possível executar os notebooks 
Jupyter individualmente.
Para fazer isso, é preciso primeiro iniciar o servidor do Jupyter
Notebook indo no diretório raiz do repositório e executando

    jupyter notebook

Isso irá iniciar o servidor e abrir o navegador padrão na interface do
Jupyter. Nessa página, é possível ir para a pasta `code/notebooks` e
selecionar o notebook que você deseja ver/executar.

Cada notebook é dividido em células (algumas possuem texto enquanto outras
possuem código).
Cada célula pode ser executada usando `Shift + Enter`.
Executar células de texto formata o texto contido nelas e executar células
de código roda o código contido nelas e produz a sua saída. 
Para executar o notebook inteiro, execute todas as células em ordem.

