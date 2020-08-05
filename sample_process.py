from SciServer import SkyQuery, Authentication

def login():
    '''Log in to SciServer'''
    return Authentication.login('******', '******')

def drop_side_or_cont_(col_names, drop):
    '''
    Called by make_sql_column_string to return a subset of column names.

    Arguments:
    - col_names: a list of column names in main data table. Contains SUBCLASS - target variable, and a number of
        local element indices each with four features:
            *cont: value of global continuum fit
            *side: value of local best fit
            *err: error term
            *mask: 0/1 value representing whether or not data at this location was good (1) or bad (0)
    - drop: subset of columns to drop - cont, side, err or mask

    Returns:
    - list of column names not containing input feature
    '''
    columns = []
    for col in col_names:
        if not drop in col:
            columns.append(col)
    return columns

def make_sql_column_string(table_source, dataset, *args):
    '''
    Return string with all column names to return with SQL query, comma separated.

    Arguments:
    - table_source: SkyServer table name to draw from
    - dataset: SkyServer dataset to draw from
    - (optional) column features to drop, if any. Can be single feature (eg 'cont') and multiple features
    '''
    try:
        columns = SkyQuery.listTableColumns(tableName = table_source, datasetName=dataset)
    except Exception as err:
        if str(err) == 'User token is not defined. First log into SciServer.':
            token = login()
            columns = SkyQuery.listTableColumns(tableName = table_source, datasetName=dataset)
        else:
            raise err
    column_names = [col['name'] for col in columns]
    if len(args) > 0:
        column_names = drop_side_or_cont_(column_names)
    id_index, subclass_index = column_names.index('SPECOBJID'), column_names.index('SUBCLASS')
    column_names[:0] = [column_names.pop(id_index), column_names.pop(subclass_index)]
    column_string = ', '.join(column_names)
    return column_string

def subsample(rows, percent, seed=42):
    '''
    Return SQL string pieces to query random subset of main data table.

    Arguments:
    - rows: number of toal row in data table
    - percent: percent of data to sample
    - seed: random seed

    Returns:
    - tuple containing SQL string pieces with 1. number of rows to return and 2. random sort syntax
    '''
    query_filter = ['','']
    if percent < 1:
        query_filter = (f'top {int(rows*percent)}', f'ORDER BY RAND({seed})')
    return query_filter
