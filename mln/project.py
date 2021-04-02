import copy
from zipfile import ZipFile, ZIP_DEFLATED
import os
import sys
from dnutils import ifnone, logs
import json
import collections
import pdb

logger = logs.getlogger(__name__)


class MLNProject(object):
    """
    Represents a .mln project archive containing MLNs, DBs and config files.
    """

    def __init__(self, name=None):
        self._name = name if name is not None and '.mln' in name else '{}.mln'.format(name or 'unknown')
        self._mlns = {}
        self.learn_conf = MLNConfig()
        self.query_conf = MLNConfig()
        self._emlns = {}
        self._dbs = {}
        self._results = {}
        self._dirty = True
        self.dirty_listeners = []

    @property
    def dirty(self):
        return self._dirty or self.learn_conf.dirty or self.query_conf.dirty

    @dirty.setter
    def dirty(self, d):
        self._dirty = d
        for l in self.dirty_listeners:
            l(d)

    def add_listener(self, listener):
        """
        add listener
        """
        self.dirty_listeners.append(listener)

    @property
    def mlns(self):
        return self._mlns

    @mlns.setter
    def mlns(self, mlns):
        self._mlns = mlns
        self.dirty = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name if name is not None and '.mln' in name else '{}.mln'.format(name or 'unknown')
        self.dirty = True

    @property
    def dbs(self):
        return self._dbs

    @dbs.setter
    def dbs(self, dbs):
        self._dbs = dbs
        self.dirty = True

    @property
    def emlns(self):
        return self._emlns

    @emlns.setter
    def emlns(self, emlns):
        self._emlns = emlns
        self.dirty = True

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        self._results = results
        self.dirty = True

    def add_mln(self, name, content=''):
        self._mlns[name] = content
        self.dirty = True

    def add_db(self, name, content=''):
        self._dbs[name] = content
        self.dirty = True

    def add_emln(self, name, content=''):
        self._emlns[name] = content
        self.dirty = True

    def add_result(self, name, content=''):
        self._results[name] = content
        self.dirty = True

    def rm_mln(self, name):
        del self._mlns[name]
        self.dirty = True

    def rm_db(self, name):
        del self._dbs[name]
        self.dirty = True

    def rm_emln(self, name):
        del self._emlns[name]
        self.dirty = True

    def rm_result(self, name):
        del self._results[name]
        self.dirty = True

    def copy(self):
        proj_ = copy.deepcopy(self)
        return proj_

    def load_mln(self, config, mln=None):
        """
        load mln
        """
        if config == 'query':
            config = self.query_conf
        elif config == 'learn':
            config = self.learn_conf
        from mln.markov_logic_network import parse_mln
        path = self.path if hasattr(self, 'path') else None
        return parse_mln(self.mlns[ifnone(mln, config['mln'])], projectpath=path, logic=config['logic'],
                         grammar=config['grammar'])

    def load_db(self, mln, config, db=None):
        """
        load database
        """
        if db is None:
            if config not in ['query', 'learn']:
                raise Exception('Need a database name or config.')
        if config == 'query':
            config = self.query_conf
        elif config == 'learn':
            config = self.learn_conf

        from mln.database import parse_db
        path = self.path if hasattr(self, 'path') else None
        return parse_db(mln, self.dbs[ifnone(db, config['db'])], ignore_unknown_preds=config['ignore_unknown_preds'],
                        projectpath=path)

    def save(self, dirpath='.'):
        filename = self.name
        self.path = dirpath
        with open(os.path.join(dirpath, filename), 'w') as zf:
            self.to_stream(zf)
        self.dirty = False

    @staticmethod
    def open(filepath):
        pdb.set_trace()
        fullpath = os.path.expanduser(os.path.expandvars(filepath))
        name = os.path.basename(fullpath)
        proj = MLNProject(name)
        proj.path = os.path.dirname(fullpath)
        #  fullpath = ZipFile(fullpath, 'w')
        with ZipFile(fullpath, 'r') as zf:
        # with open(fullpath) as zf:
            for member in zf.namelist():
                if member == 'learn.conf':
                    tmpconf = eval(zf.open(member).read())
                    proj.learn_conf = MLNConfig()
                    proj.learn_conf.update(tmpconf)
                elif member == 'query.conf':
                    tmpconf = eval(zf.open(member).read())
                    proj.query_conf = MLNConfig()
                    proj.query_conf.update(tmpconf)
                else:
                    path, f = os.path.split(member)
                    if path == 'mlns':
                        proj._mlns[f] = zf.open(member).read()
                    elif path == 'emlns':
                        proj._emlns[f] = zf.open(member).read()
                    elif path == 'dbs':
                        proj._dbs[f] = zf.open(member).read()
                    elif path == 'results':
                        proj._results[f] = zf.open(member).read()
        return proj

    @staticmethod
    def read(filename, stream, path=None):
        proj = MLNProject(filename)
        proj.path = path
        with ZipFile(stream, 'r') as zf:
            for member in zf.namelist():
                if member == 'learn.conf':
                    tmpconf = eval(zf.open(member).read())
                    proj.learn_conf = MLNConfig()
                    proj.learn_conf.update(tmpconf)
                elif member == 'query.conf':
                    tmpconf = eval(zf.open(member).read())
                    proj.query_conf = MLNConfig()
                    proj.query_conf.update(tmpconf)
                else:
                    path, f = os.path.split(member)
                    if path == 'mlns':
                        proj._mlns[f] = zf.open(member).read()
                    elif path == 'emlns':
                        proj._emlns[f] = zf.open(member).read()
                    elif path == 'dbs':
                        proj._dbs[f] = zf.open(member).read()
                    elif path == 'results':
                        proj._results[f] = zf.open(member).read()
        return proj

    def to_stream(self, stream):
        with ZipFile(stream, 'w', ZIP_DEFLATED) as zf:
            # save the learn.conf
            zf.writestr('learn.conf', self.learn_conf.dumps())
            # save the query.conf
            zf.writestr('query.conf', self.query_conf.dumps())
            # save the MLNs
            for name, mln in self.mlns.iteritems():
                zf.writestr(os.path.join('mlns', name), mln)
            # save the model extensions
            for name, emln in self.emlns.iteritems():
                zf.writestr(os.path.join('emlns', name), emln)
            # save the DBs
            for name, db in self.dbs.iteritems():
                zf.writestr(os.path.join('dbs', name), db)
            # save the results
            for name, result in self.results.iteritems():
                zf.writestr(os.path.join('results', name), result)

    def write(self, stream=sys.stdout):
        stream.write('MLN Project: %s\n' % self.name)
        if self.learn_conf is not None:
            stream.write('learn.conf\n')
        if self.query_conf is not None:
            stream.write('query.conf\n')
        stream.write('mlns/\n')
        for name in self.mlns:
            stream.write('  ./%s\n' % name)
        stream.write('dbs/\n')
        for name in self.dbs:
            stream.write('  ./%s\n' % name)
        stream.write('emlns/\n')
        for name in self.emlns:
            stream.write('  ./%s\n' % name)
        stream.write('results/\n')
        for name in self.results:
            stream.write('  ./%s\n' % name)


def convert(data):
    """
    convert everything to ASCII
    """
    if isinstance(data, str):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


class MLNConfig(object):

    def __init__(self, filepath=None):
        self.config_file = MLNPath(filepath) if filepath is not None else None
        self.config = {}
        self._dirty = False
        if self.config_file is None or not self.config_file.exists:
            self.config = {}
        else:
            if self.config_file.project is not None:
                self.config = convert(dict(self.config_file.content.config))
            else:
                self.config = convert(json.loads(self.config_file.content))
            logger.debug('loaded %s config' % self.config_file.compose())

    @property
    def dirty(self):
        return self._dirty

    def get(self, k, d=None):
        return self.config.get(k, d)

    def update(self, d):
        self.config.update(d)
        self._dirty = True

    def __getitem__(self, s):
        if type(s) is slice:
            prim = s.start
            sec = s.stop
            if self.config.get(prim) is not None:
                return self.config.get(prim).get(sec)
            else:
                return None
        else:
            return self.config.get(s)

    def __setitem__(self, s, v):
        if type(s) is slice:
            prim = s.start
            sec = s.stop
            p = self.config.get(prim)
            if p is None:
                p = {}
                self.config[prim] = p
            p[sec] = v
        else:
            self.config[s] = v
        self._dirty = True

    def dump(self):
        if self.config_file is None:
            raise Exception('no filename specified')
        if self.config_file.project is not None:
            project = MLNPath(self.config_file.projectloc).content
            if self.config_file.file == 'query.conf':
                project.query_conf.config = self.config
            elif self.config_file.file == 'learn.conf':
                project.query_conf.file = self.config
            else:
                Exception('Invalid config file name: %s' % self.config_file.file)
            project.save(project.resolve_path())
        else:
            with open(os.path.join(self.config_file.resolve_path(), self.config_file.file), 'w+') as cf:
                cf.write(json.dumps(self.config))
        self._dirty = False

    def dumps(self):
        self._dirty = False
        return json.dumps(self.config, indent=4)


class MLNPath(object):
    """
    Loads the MLN resource content from a location.

    A location can be a regular absolute or relative path to an `.mln` file. It may also refer
    to an MLN inside a `.mln` project container. In the latter case, the `.mln` file name
    needs to be separated from the `.mln` project location by a colon. Path specification
    may also contain references to system environment variables, which are referred to of the
    form ``${var}``.

    :Example:

    >>> MLNPath('~/mlns/classification.mln:model-1.mln').content
    ...


    """

    def __init__(self, path):

        # split the path wrt slashes
        self.path, file = os.path.split(path)

        self._abspath = self.path.startswith('/')
        if ':' in file or file.endswith('.mln'):
            res = file.split(':')
            if len(res) == 2:
                self.project, self.file = res
            elif len(res) == 1:
                self.project, self.file = res[0], None
        else:
            self.project = None
            self.file = file

    def compose(self):
        p = os.path.join(*self.path)
        if self.project is not None:
            p += ('/' if p else '') + self.project
            if self.file is not None:
                p += ':' + str(self.file)
        else:
            p += ifnone(self.file, '', lambda x: '/' + str(x))
        return p

    def resolve_path(self):
        p = self.path
        for f in (os.path.expanduser, os.path.expandvars, os.path.normpath):
            p = f(p)
        return p

    @property
    def file(self):
        """
        Returns the name of the file specified by this ``MLNPath``.
        """
        return self._file

    @file.setter
    def file(self, f):
        self._file = f

    @property
    def project(self):
        """
        Returns the project name specified.
        """
        return self._project

    @project.setter
    def project(self, p):
        self._project = p

    @property
    def content(self):
        """
        Returns the content of the file specified by this ``MLNPath``.
        """
        path = self.resolve_path()
        if self.project is not None:
            proj = MLNProject.open(os.path.join(self.resolve_path(), self.project))
            if self.file is None:
                return proj
            fileext = self.file.split('.')[-1]
            if fileext == 'mln':
                mln = proj.mlns.get(self.file)
                if mln is None:
                    raise Exception(
                    'Project %s does not contain and MLN named %s' % (self.project, self.file))
                return mln
            elif fileext == 'db':
                db = proj.dbs.get(self.file)
                if db is None: raise Exception(
                    'Project %s does not contain a database named %s' % (self.project, self.file))
                return db
            elif fileext == 'conf':
                conf = {'query.conf': proj.query_conf, 'learn.conf': proj.learn_conf}.get(self.file)
                if conf is None: raise Exception(
                    'Project %s does not contain a config file named %s' % (self.project, self.file))
                return conf
        else:
            with open(os.path.join(path, self.file)) as f:
                return f.read()

    @property
    def projectloc(self):
        """
        Returns the location of the project file, if any is specified.
        """
        if self.project is None:
            raise Exception('No project specified in the path.')
        return os.path.join(self.resolve_path(), self.project)

    @property
    def exists(self):
        """
        Checks if the file exists.
        """
        return os.path.exists(os.path.join(self.resolve_path(), ifnone(self.project, self.file)))

    @property
    def isabs(self):
        return self._abspath

    def __str__(self):
        return self.compose()

    def __repr__(self):
        return 'MLNPath(%s)' % str(self)


