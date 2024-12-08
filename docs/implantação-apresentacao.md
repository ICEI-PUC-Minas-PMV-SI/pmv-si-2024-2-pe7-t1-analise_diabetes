# Implantação da solução

<!--Nesta seção, a implantação da solução proposta em nuvem deverá ser realizada e detalhadamente descrita. Além disso, deverá ser descrito também, o planejamento da capacidade operacional através da modelagem matemática e da simulação do sistema computacional.

Após a implantação, realize testes que mostrem o correto funcionamento da aplicação. -->
## API

Para implantação da API que irá gerênciar os modelos devemos primeiro garantir que o código disponível em [src/api](/scr/api) deve estar versionado em um repositório como GitHub, GitLab ou Bitbucket. <br/><br/>
Os passos a seguir dizem respeito a implantação em uma cloud AWS, porém para outros provedores de serviços cloud a mesma lógica é aplicável.

1 -  Configurar o Ambiente Local

Certifique-se de que o arquivo app.py inicia o servidor na porta correta (geralmente 5000): <br/>

```
if __name__ == "__main__":
app.run(host="0.0.0.0", port=5000)
```

<br/>

2 - Criar um Arquivo ZIP

Inclua a pasta src/api, o requirements.txt.
Certifique-se de que os modelos estão no diretório correto ou podem ser acessados.

<br/>

3 - Implantar no Elastic Beanstalk

No console da AWS, vá para Elastic Beanstalk e crie um novo ambiente.
Escolha Python como a plataforma.
Faça upload do arquivo ZIP e inicie o ambiente.

## FRONTEND

Para implantação do frontend que irá garantir o forumlário pro usuário final devemos primeiro garantir que o código disponível em [src/frontend](/scr/frontend) deve estar versionado em um repositório como GitHub, GitLab ou Bitbucket. <br/><br/>

1 - Configurar o AWS Amplify
 . Acesse o Console do Amplify 
 . Clique em Get Started em "Host your web app".
 . Escolha o provedor de Git e conecte seu repositório.
 . Escolha a branch que deseja implantar (geralmente main ou master).

2 - Configuração de Build
. O AWS Amplify detectará automaticamente que sua aplicação é baseada em Vite e fornecerá uma configuração padrão. Verifique se o arquivo amplify.yml inclui: <br/>

```
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm install
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: dist
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
```
<br/>
. Certifique-se de que o baseDirectory aponta para dist, que é o diretório gerado pelo comando npm run build.`

3 - Inicie o deploy clicando em "Save and Deploy".




# Apresentação da solução

No Link abaixo você pode conferir a demonstração de funcionalidade da solução final: <br/>
[Link para Demo do projeto](https://drive.google.com/file/d/1_AtAd2CNUHjQ2A_dXcsG3zslaOm76ls-/view?usp=sharing).

