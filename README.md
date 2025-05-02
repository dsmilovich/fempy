# TP 3
El TP tiene como objetivo la implementacion en Python de formas debiles y solvers (esquemas de solucion) para la solucion de la ecuacion del calor transitoria sobre el codigo facilitado.
El codigo se estructura como un proyecto en un unico archivo, con una funcion `main` que define la malla, el espacio de funciones, el problema a resolver, las condiciones de borde, el tipo de solver, y el esquema de solucion (transitorio, estacionario, etc.).
Los objetos anteriormente mencionados se encuentran implementados como clases en Python.
El codigo se enmarca dentro del paradigma de la programacion orientada a objectos. Se sugiere ver [material complementario relacionado a OOP y Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/).

# Prerrequisitos
Los requisitos basicos para poder interactuar con el codigo son
## Git Bash
Descargar de [este enlace](https://git-scm.com/downloads) con las opciones por defecto.
Esta herramienta permite trabajar con una terminal de [bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)).
## Python
Descargar de [este enlace](https://www.python.org/downloads/release/python-3133/), 32 o 64 bits segun corresponda.
Iniciar Git Bash y ejecutar

`$ which python`

lo cual deberia devolver la ruta donde se instalo Python.
### matplotlib
En una terminal de Git Bash con Python instalado, ejecutar

`$ pip install matplotlib`

# Uso
## Clonar el repositorio
Abrir Git Bash y crear un directorio de trabajo, por ejemplo

`$ mkdir tp3`

Luego moverse al directorio creado y clonar el repositorio

```
$ cd tp3
$ git clone https://github.com/dsmilovich/fempy.git
```

## Ejecutar el codigo
Con el repositorio clonado, ejecutar
```
$ cd fempy
$ python fempy.py
```
# Modalidad de trabajo
Se sugiere crear y trabajar en un repositorio particular en [github](https://github.com/).
Una vez creado el repositorio, subir el codigo proporcionado.
El entregable del TP 3 es un repositorio con un codigo funcional donde todos los requerimientos planteados han sido implementados y testeados.

# Requerimientos
