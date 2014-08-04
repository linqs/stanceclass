package edu.ucsc.cs

import java.util.Set;
import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.ucsc.cs.utils.Evaluator;
import edu.ucsc.cs.utils.ResultWriter;



//dataSet = "fourforums"
dataSet = "stance-classification"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

subDir = args[1]
fold = args[2]
def dir = 'data'+java.io.File.separator + subDir + java.io.File.separator + fold + java.io.File.separator + 'train' + java.io.File.separator;
def testdir = 'data'+java.io.File.separator + subDir + java.io.File.separator + fold + java.io.File.separator + 'test' + java.io.File.separator;

initialWeight = 5

PSLModel model = new PSLModel(this, data)

/* 
 * List of predicates with their argument types
 * writesPost(Author, Post) -- observed
 * participatesIn(Author, Topic) -- observed
 * hasTopic(Post, Topic) -- observed
 * isProAuth(Author, Topic) -- target
 * isProPost(Post, Topic) -- target
 * agreesAuth(Author, Author) -- observed 
 * agreesPost(Post, Post) -- observed
 * hasLabelPro(Post, Topic) -- observed
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "hasLabelPro" , types:[ArgumentType.UniqueID, ArgumentType.String]

model.add rule : (hasLabelPro(A, T)) >> isProAuth(A, T) , weight : initialWeight
model.add rule : (~(hasLabelPro(A, T))) >> ~(isProAuth(A, T)) , weight : initialWeight

/*
 * Inserting data into the data store
 */

Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition observed_te = new Partition(3);
Partition predict_te = new Partition(4);
Partition truth_te = new Partition(5);
Partition dummy_tr = new Partition(6);
Partition dummy_te = new Partition(7);

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"hasLabelPro.csv", ",");

/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(isProAuth, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"isProAuth.csv", ",");

/*
 * Testing split for model inference
 * Observed partitions
 */

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"hasLabelPro.csv", ",");

/*
 * Label partitions
 */

inserter = data.getInserter(isProAuth, truth_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"isProAuth.csv", ",");

/*
 * Set up training databases for weight learning
 */

Database distributionDB = data.getDatabase(predict_tr, [hasLabelPro] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProAuth] as Set)

/* Populate isProAuth in observed DB. */
DatabasePopulator populator = new DatabasePopulator(distributionDB);
populator.populateFromDB(truthDB, isProAuth);

Database testDB = data.getDatabase(predict_te, [hasLabelPro] as Set, observed_te);
Database testTruthDB = data.getDatabase(truth_te, [isProAuth] as Set)

/* Populate isProAuth in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(testTruthDB, isProAuth);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference()

Evaluator evaluator = new Evaluator(testDB, isProAuth, "authorstance_baseline", fold);
evaluator.outputToFile();

/* Accuracy */
def discComp = new DiscretePredictionComparator(testDB)
discComp.setBaseline(testTruthDB)
discComp.setResultFilter(new MaxValueFilter(isProAuth, 1))
discComp.setThreshold(0.5) // treat best value as true as long as it is nonzero

Set<GroundAtom> groundings = Queries.getAllAtoms(testTruthDB, isProAuth)
int totalTestExamples = groundings.size()
DiscretePredictionStatistics stats = discComp.compare(isProAuth, totalTestExamples)
System.out.println("Accuracy: " + stats.getAccuracy())
accuracy = (double) stats.getAccuracy()

/* Evaluation */

def comparator = new SimpleRankingComparator(testDB)
comparator.setBaseline(testTruthDB)

// Choosing what metrics to report
def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
double [] score = new double[metrics.size() + 1]

try {
    for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(isProAuth)
    }
    score[metrics.size()] = accuracy
    //Storing the performance values of the current fold
    System.out.println(fold + "," + score[0] + "," + score[1] + "," + score[2])
    
    
    ResultWriter rs = new ResultWriter(score, fold, 'result_baseline.txt')
    rs.write()
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}

testTruthDB.close()
testDB.close()
distributionDB.close()
truthDB.close()
