runfile('E:/mao/model.py', wdir='E:/mao')
Traceback (most recent call last):

  File "<ipython-input-3-36417596cd13>", line 1, in <module>
    runfile('E:/mao/model.py', wdir='E:/mao')

  File "C:\Program Files\Anaconda3\envs\tensorflow\lib\site-packages\spyder\utils\site\sitecustomize.py", line 710, in runfile
    execfile(filename, namespace)

  File "C:\Program Files\Anaconda3\envs\tensorflow\lib\site-packages\spyder\utils\site\sitecustomize.py", line 101, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "E:/mao/model.py", line 152, in <module>
    main()

  File "E:/mao/model.py", line 118, in main
    md = ModelData(train=train_data, test=test_data, cont_vars=['503','504'], cat_vars=['501','502'])

  File "E:/mao/model.py", line 28, in __init__
    self.category_sizes = [(v,len(train[v].cat.categories)) for v in cat_vars] # cardinality of each category

  File "E:/mao/model.py", line 28, in <listcomp>
    self.category_sizes = [(v,len(train[v].cat.categories)) for v in cat_vars] # cardinality of each category

  File "C:\Program Files\Anaconda3\envs\tensorflow\lib\site-packages\pandas\core\generic.py", line 3610, in __getattr__
    return object.__getattribute__(self, name)

  File "C:\Program Files\Anaconda3\envs\tensorflow\lib\site-packages\pandas\core\accessor.py", line 54, in __get__
    return self.construct_accessor(instance)

  File "C:\Program Files\Anaconda3\envs\tensorflow\lib\site-packages\pandas\core\categorical.py", line 2211, in _make_accessor
    raise AttributeError("Can only use .cat accessor with a "

AttributeError: Can only use .cat accessor with a 'category' dtype
