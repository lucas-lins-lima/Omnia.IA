# Estrutura de Diretórios do Projeto Omnia.IA

Este documento detalha a organização e o propósito de cada pasta e arquivo dentro da estrutura de diretórios do projeto Omnia.IA. Compreender essa estrutura é essencial para navegar e contribuir para o desenvolvimento da plataforma.
```
ia_generativa/
│
├── main.py             # Ponto de entrada da aplicação FastAPI
├── requirements.txt    # Lista de dependências do Python
├── Dockerfile          # Arquivo de configuração do Docker
├── .env.example        # Arquivo de exemplo para variáveis de ambiente
│
├── config/             # Configurações da aplicação
│   └── settings.py     # Configurações de modelos, paths, etc.
│
├── models/             # Lógica de carregamento e inferência dos modelos de IA
│   ├── base_model.py   # Classe base para modelos (opcional)
│   ├── text_generator.py  # Lógica para modelos de texto (LLMs)
│   ├── image_generator.py # Lógica para modelos de imagem (Stable Diffusion)
│   ├── audio_processor.py # Lógica para processamento de áudio (Whisper, Bark)
│   └── video_processor.py # Lógica para processamento de vídeo
│
├── preprocessors/      # Módulos para pré-processamento de diferentes tipos de mídia
│   ├── pdf_processor.py   # Extração de texto/imagens de PDFs
│   ├── xlsx_processor.py  # Leitura de dados de arquivos XLSX
│   ├── image_processor.py # Redimensionamento, normalização de imagens
│   ├── audio_processor.py # Conversão, resampling de áudio
│   └── video_processor.py # Extração de frames/áudio de vídeos
│
├── api/                # Definição da API FastAPI
│   ├── v1/               # Versão 1 da API
│   │   ├── endpoints.py # Definição das rotas da API e lógica de requisição/resposta
│   │   └── schemas.py   # Modelos Pydantic para validação de dados
│
├── utils/              # Funções utilitárias e helpers
│   ├── file_utils.py    # Funções para manipulação de arquivos
│   ├── logging_config.py# Configuração centralizada de logging
│   └── orchestrator.py  # Lógica de roteamento e coordenação de tarefas
│
├── tests/              # Diretório para testes unitários e de integração
│   ├── unit/
│   └── integration/
│
└── data/               # Diretório para dados temporários e modelos baixados
├── models_cache/   # Cache para pesos de modelos do HuggingFace
└── temp_uploads/   # Armazenamento temporário de arquivos enviados
```
**Arquivos e Pastas na Raiz (`ia_generativa/`)**:

* **`main.py`**: Este é o ponto de entrada principal da aplicação construída com o framework FastAPI. Ao executar este arquivo, o servidor da sua API será iniciado. Ele provavelmente contém a inicialização da instância do FastAPI e a inclusão das rotas definidas no diretório `api`.

* **`requirements.txt`**: Este arquivo lista todas as bibliotecas Python das quais o projeto depende para funcionar corretamente. Ele é utilizado pela ferramenta `pip` para instalar as dependências no seu ambiente de desenvolvimento ou no container Docker.

* **`Dockerfile`**: Este arquivo contém as instruções para construir uma imagem Docker da sua aplicação. O Docker permite criar um ambiente isolado e reproduzível para a execução da sua IA, garantindo que todas as dependências e configurações estejam corretas.

* **`.env.example`**: Este arquivo serve como um modelo para o arquivo `.env`, onde você definirá as variáveis de ambiente específicas da sua configuração (por exemplo, chaves de API, configurações de banco de dados, etc.). É uma boa prática incluir um arquivo de exemplo para que outros colaboradores saibam quais variáveis precisam ser configuradas. O arquivo `.env` em si geralmente não é versionado para evitar a exposição de informações sensíveis.

**Pasta `config/`**:

* Esta pasta é dedicada a armazenar arquivos de configuração da sua aplicação.
    * **`settings.py`**: Este arquivo provavelmente conterá configurações globais para a sua aplicação, como paths para diretórios de modelos pré-treinados, configurações específicas para cada modelo de IA, variáveis de controle de fluxo e outras definições que podem precisar ser ajustadas.

**Pasta `models/`**:

* Esta pasta contém o código responsável pelo carregamento dos modelos de inteligência artificial e pela execução da inferência (o processo de gerar resultados a partir dos modelos). Cada arquivo dentro desta pasta representa a lógica para um tipo específico de modelo.
    * **`base_model.py`**: Este arquivo pode conter uma classe base ou interfaces comuns que são compartilhadas entre os diferentes modelos de IA. Isso ajuda a manter a consistência e facilita a extensão do sistema com novos modelos no futuro. (Opcional, pode não existir inicialmente).
    * **`text_generator.py`**: Contém a lógica para carregar e utilizar os modelos de linguagem de grande escala (LLMs) para tarefas como geração de texto, resumo, tradução, etc. Modelos como Llama 3, Mistral, Falcon e GPT-NeoX seriam gerenciados aqui.
    * **`image_generator.py`**: Contém o código para carregar e executar modelos generativos de imagem, como Stable Diffusion. Também pode incluir a lógica para outros modelos relacionados a imagens, como os de legendagem (BLIP-2, CLIP) e edição (ControlNet).
    * **`audio_processor.py`**: Contém a lógica para modelos de processamento de áudio, incluindo speech-to-text (como Whisper), text-to-speech (como Bark e VITS) e geração de música (como MusicGen).
    * **`video_processor.py`**: Contém o código para lidar com modelos de processamento de vídeo. Inicialmente, pode incluir lógica para processar vídeos frame por frame usando os modelos de imagem ou para tarefas de compreensão de vídeo (usando modelos como X-CLIP ou LVT). Modelos emergentes de geração de vídeo como AnimateDiff ou Stable Video Diffusion também seriam integrados aqui.

**Pasta `preprocessors/`**:

* Esta pasta contém módulos responsáveis por preparar os dados de entrada para os modelos de IA. Cada arquivo aqui lida com o pré-processamento de um tipo específico de mídia.
    * **`pdf_processor.py`**: Contém funções para extrair informações de arquivos PDF, como texto, imagens e tabelas. Pode incluir a integração com ferramentas de Optical Character Recognition (OCR) como Tesseract para processar PDFs digitalizados.
    * **`xlsx_processor.py`**: Contém o código para ler e manipular dados de arquivos XLSX (planilhas eletrônicas) usando bibliotecas como `pandas` e `openpyxl`. Isso pode envolver a identificação de cabeçalhos, a conversão de tipos de dados e o tratamento de valores ausentes.
    * **`image_processor.py`**: Contém funções para realizar operações de pré-processamento em arquivos de imagem (PNG, JPEG, etc.), como redimensionamento, normalização de valores de pixel e conversão de espaços de cor, conforme exigido pelos modelos de imagem.
    * **`audio_processor.py`**: Contém funções para pré-processar arquivos de áudio (como MP3), incluindo conversão de formato (para WAV, por exemplo), ajuste da taxa de amostragem (resampling), normalização de volume e, potencialmente, extração de características como espectrogramas ou MFCCs.
    * **`video_processor.py`**: Contém o código para pré-processar arquivos de vídeo. Isso pode envolver a extração de quadros individuais (frames) do vídeo e a separação da trilha de áudio para processamento independente.

**Pasta `api/`**:

* Esta pasta define a interface da sua aplicação através de uma API (Application Programming Interface) construída com o framework FastAPI. Ela permite que outros sistemas ou usuários interajam com a sua IA.
    * **`v1/`**: Esta subpasta representa a primeira versão da sua API. Organizar a API por versões (por exemplo, `v1`, `v2`) é uma prática comum para manter a compatibilidade com clientes existentes ao introduzir novas funcionalidades ou alterações na API.
        * **`endpoints.py`**: Este arquivo contém a definição das rotas da API (os URLs que os clientes podem acessar) e a lógica de tratamento para cada requisição. Por exemplo, ele definiria os endpoints para gerar texto, processar imagens, etc., e conteria as funções que são executadas quando esses endpoints são acessados.
        * **`schemas.py`**: Este arquivo define os modelos de dados utilizando a biblioteca Pydantic. Esses modelos são usados para validar a estrutura e os tipos dos dados que são recebidos pela API (na requisição) e enviados como resposta, garantindo a consistência e a integridade dos dados.

**Pasta `utils/`**:

* Esta pasta contém módulos com funções utilitárias e helpers que são usados em diferentes partes da aplicação para realizar tarefas comuns.
    * **`file_utils.py`**: Pode conter funções para manipulação de arquivos, como leitura, escrita, exclusão, verificação de tipos, etc.
    * **`logging_config.py`**: Este arquivo provavelmente conterá a configuração centralizada para o sistema de logging da aplicação. O logging é importante para registrar eventos, erros e informações de depuração, facilitando o monitoramento e a identificação de problemas.
    * **`orchestrator.py`**: Contém a lógica do "Núcleo Orquestrador" mencionado na visão geral do projeto. Ele é responsável por gerenciar o fluxo de processamento, recebendo as requisições, identificando o tipo de mídia, roteando para os módulos de pré-processamento e modelos apropriados e coordenando a execução das tarefas.

**Pasta `tests/`**:

* Esta pasta é dedicada a conter os testes automatizados para a sua aplicação. Os testes são cruciais para garantir que o código funcione corretamente e para prevenir a introdução de erros em futuras modificações.
    * **`unit/`**: Contém os testes unitários, que focam em testar pequenas unidades de código isoladamente (por exemplo, funções ou métodos específicos).
    * **`integration/`**: Contém os testes de integração, que verificam como diferentes partes da aplicação (por exemplo, módulos interagindo entre si) funcionam em conjunto.

**Pasta `data/`**:

* Esta pasta é usada para armazenar dados temporários e outros arquivos relacionados ao funcionamento da aplicação que não fazem parte do código fonte principal.
    * **`models_cache/`**: Este diretório serve como um cache local para os pesos dos modelos de IA baixados da plataforma HuggingFace ou de outras fontes. Isso evita a necessidade de baixar os modelos repetidamente, economizando tempo e largura de banda.
    * **`temp_uploads/`**: Este diretório é usado para armazenar temporariamente os arquivos que são enviados para a API para processamento. Os arquivos podem ser armazenados aqui durante o pré-processamento e a inferência e, em seguida, excluídos.

Compreender a função de cada pasta e arquivo nesta estrutura de diretórios facilitará o desenvolvimento, a manutenção e a colaboração no projeto Omnia.IA.
