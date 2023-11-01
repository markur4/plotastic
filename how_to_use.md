```python
import plotastic as plst
```

## Import Example Data


```python
DF, dims = plst.load_dataset("qpcr")
```

    #! Imported seaborn dataset 'qpcr' 
    	 columns:Index(['Unnamed: 0', 'gene', 'method', 'fraction', 'subject', 'donor', 'uFC',
           'class', 'FC'],
          dtype='object')
    	 dimensions: {'y': 'FC', 'x': 'gene', 'hue': 'fraction', 'col': 'class', 'row': 'method'}



```python
DF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>gene</th>
      <th>method</th>
      <th>fraction</th>
      <th>subject</th>
      <th>donor</th>
      <th>uFC</th>
      <th>class</th>
      <th>FC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>IFNG</td>
      <td>MACS</td>
      <td>F1</td>
      <td>1</td>
      <td>3266</td>
      <td>0.003071</td>
      <td>ECM &amp; Adhesion</td>
      <td>1.036131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>IFNG</td>
      <td>MACS</td>
      <td>F1</td>
      <td>3</td>
      <td>7613</td>
      <td>0.003005</td>
      <td>ECM &amp; Adhesion</td>
      <td>1.013966</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>IFNG</td>
      <td>MACS</td>
      <td>F1</td>
      <td>4</td>
      <td>9721</td>
      <td>0.002762</td>
      <td>ECM &amp; Adhesion</td>
      <td>0.932101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>IFNG</td>
      <td>MACS</td>
      <td>F1</td>
      <td>5</td>
      <td>9526</td>
      <td>0.002922</td>
      <td>ECM &amp; Adhesion</td>
      <td>0.986034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>IFNG</td>
      <td>MACS</td>
      <td>F2</td>
      <td>1</td>
      <td>3266</td>
      <td>0.008619</td>
      <td>ECM &amp; Adhesion</td>
      <td>2.908520</td>
    </tr>
  </tbody>
</table>
</div>




```python
dims
```




    {'y': 'FC', 'x': 'gene', 'hue': 'fraction', 'col': 'class', 'row': 'method'}



## Initialize DataAnalysis Object
- DataAnalysis will give you feedback on missing data:
  - NaN count
  - Samplesize




```python
DA = plst.DataAnalysis(data=DF,           # Dataframe
                       dims=dims,         # Dictionary with y, x, hue, col, row 
                       subject="subject", # Data is paired by subject (optional)
                       )
```

    ================================================================================
    #! checking data integrity...
    ❗️ DATA INCOMPLETE: Among all combinations of levels from selected factors ['method', 'class', 'fraction', 'gene'], groups/facets are missing in the Dataframe. 👉 Call .data_get_empty_groupkeys() to see them all.
    ✅ GROUPS COMPLETE: No groups with NaNs.
    🫠 GROUPS UNEQUAL: Groups (114 total) have different samplesizes (n = 5.2 ±0.89). 👉 Call .data_get_samplesizes() to see them.
    These are the 5 groups with the smallest samplesizes:
    method  class            fraction  gene 
    MACS    Bone Metabolism  F1        SOST     4
                                       FBN1     4
            MMPs             F1        CCL20    3
            Bone Metabolism  F1        TIMP1    3
            MMPs             F1        IL2RG    1
    Name: FC, dtype: int64
     🌳 LEVELS WELL CONNECTED: These Factors have levels that are always found together: ['method', 'fraction']. Call .levels_combocount() or .levels_dendrogram() to see them all.
    ❗️ Subjects incomplete: The largest subject contains 57 datapoints, but these subjects contain less:
    subject
    3     56
    4     56
    5     56
    6     56
    12    56
    8     55
    1     54
    2     38
    Name: FC, dtype: int64
    ================================================================================



```python
DA.levels_dendrogram
```




    <bound method DimsAndLevels.levels_dendrogram of <plotastic.dataanalysis.dataanalysis.DataAnalysis object at 0x2816b5cd0>>


