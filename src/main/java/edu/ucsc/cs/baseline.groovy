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
import edu.ucsc.cs.utils.Evaluator;

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator



//dataSet = "fourforums"
dataSet = "stance-classification"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

PSLModel model = new PSLModel(this, data)

/*
 * Author predicates of the form: predicate(authorID, authorID, topic) 
 * or (authorID, topic) 
 * or (authorID, postID)
 * Observed predicates
 */

/*
 * Post level observed predicates
 */

model.add predicate: "hasTopic" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "hasLabelPro" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "hasLabelAnti" , types:[ArgumentType.UniqueID, ArgumentType.String]

/*
 * Auxiliary topic predicate
 */
model.add predicate: "topic" , types:[ArgumentType.String]

/*
 * Target predicates
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isAntiAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "isProPost" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isAntiPost" , types:[ArgumentType.UniqueID, ArgumentType.String]



model.add rule : (hasLabelPro(P, T) ) >> isProPost(P, T) , weight : 1
model.add rule : (hasLabelAnti(P, T) ) >> isAntiPost(P, T) , weight : 1

model.add rule : isProPost(P, T) >> ~isAntiPost(P, T) , constraint: true
model.add rule: (~(isAntiPost(P, T)) & hasTopic(P, T)) >> isProPost(P, T), constraint:true


/*
 * Inserting data into the data store
 */
//fold = 1

//foldStr = "fold" + String.valueOf(fold) + java.io.File.separator;

/* training partitions */
Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition dummy_tr = new Partition(3);

/*testing partitions */
Partition observed_te = new Partition(4);
Partition predict_te = new Partition(5);
Partition dummy_te = new Partition(6);

/*separate partitions for the gold standard truth for testing */
Partition postProTruth = new Partition(7);
Partition postAntiTruth = new Partition(8);
Partition authProTruth = new Partition(9);
Partition authAntiTruth = new Partition(10);


//def dir = 'data'+java.io.File.separator+ foldStr + 'train'+java.io.File.separator;
def dir = 'data'+java.io.File.separator+ 'stance-dev'+java.io.File.separator + 'train'+java.io.File.separator;

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"prolabels.csv", ",");

inserter = data.getInserter(hasLabelAnti, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"antilabels.csv", ",");

inserter = data.getInserter(hasTopic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"post_topics.csv", ",");

inserter = data.getInserter(topic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"topics.csv", ",");


/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(isProPost, truth_tr)
InserterUtils.loadDelimitedData(inserter, dir+"post_pro.csv",",");

inserter = data.getInserter(isProAuth, truth_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authorpro.csv", ",");

inserter = data.getInserter(isAntiPost, truth_tr)
InserterUtils.loadDelimitedData(inserter, dir+"post_anti.csv",",");

inserter = data.getInserter(isAntiAuth, truth_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authoranti.csv", ",");


/*db population for all possible stance atoms*/

inserter = data.getInserter(isProAuth, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "participates.csv", ",")

inserter = data.getInserter(isAntiAuth, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "participates.csv", ",")

inserter = data.getInserter(isProPost, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "post_topics.csv", ",")

inserter = data.getInserter(isAntiPost, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "post_topics.csv", ",")

/*
 * Testing split for model inference
 * Observed partitions
 */

//def testdir = 'data'+java.io.File.separator+ foldStr + 'test'+java.io.File.separator;
def testdir = 'data'+java.io.File.separator+ 'stance-dev' +java.io.File.separator+ 'test'+java.io.File.separator;

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"prolabels.csv", ",");

inserter = data.getInserter(hasLabelAnti, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"antilabels.csv", ",");

inserter = data.getInserter(hasTopic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"post_topics.csv", ",");

inserter = data.getInserter(topic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"topics.csv",",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(isProPost, postProTruth)
InserterUtils.loadDelimitedData(inserter, testdir+"post_pro.csv",",");

inserter = data.getInserter(isProAuth, authProTruth)
InserterUtils.loadDelimitedData(inserter, testdir+"authorpro.csv", ",");

inserter = data.getInserter(isAntiPost, postAntiTruth)
InserterUtils.loadDelimitedData(inserter, testdir+"post_anti.csv",",");

inserter = data.getInserter(isAntiAuth, authAntiTruth)
InserterUtils.loadDelimitedData(inserter, testdir+"authoranti.csv", ",");

/*to populate testDB with the correct rvs */
inserter = data.getInserter(isProAuth, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "participates.csv", ",")

inserter = data.getInserter(isAntiAuth, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "participates.csv", ",")

inserter = data.getInserter(isProPost, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "post_topics.csv", ",")

inserter = data.getInserter(isAntiPost, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "post_topics.csv", ",")


/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [hasLabelPro, hasLabelAnti, hasTopic, topic] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProPost, isProAuth, isAntiAuth, isAntiPost] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [isProAuth, isAntiAuth, isProPost, isAntiPost] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, isProPost);
dbPop.populateFromDB(dummy_DB, isAntiPost);
dbPop.populateFromDB(dummy_DB, isProAuth);
dbPop.populateFromDB(dummy_DB, isAntiAuth);

/*
HardEM weightLearning = new HardEM(model, distributionDB, truthDB, cb);
println "about to start weight learning"
weightLearning.learn();
println " finished weight learning "
weightLearning.close();
*/
/*
 MaxPseudoLikelihood mple = new MaxPseudoLikelihood(model, trainDB, truthDB, cb);
 println "about to start weight learning"
 mple.learn();
 println " finished weight learning "
 mlpe.close();
 */

println model;

Database testDB = data.getDatabase(predict_te, [hasLabelPro, hasLabelAnti, hasTopic, topic] as Set, observed_te);

Database testTruth_postPro = data.getDatabase(postProTruth, [isProPost] as Set)
Database testTruth_postAnti = data.getDatabase(postAntiTruth, [isAntiPost] as Set)
Database testTruth_authPro = data.getDatabase(authProTruth, [isProAuth] as Set)
Database testTruth_authAnti = data.getDatabase(authAntiTruth, [isAntiAuth] as Set)

Database dummy_test = data.getDatabase(dummy_te, [isProAuth, isAntiAuth, isProPost, isAntiPost] as Set)

/* Populate in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(dummy_test, isProAuth);
test_populator.populateFromDB(dummy_test, isProPost);

test_populator.populateFromDB(dummy_test, isAntiPost);
test_populator.populateFromDB(dummy_test, isAntiAuth);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference()
System.out.println("Objective: " + result.getTotalWeightedIncompatibility())


Evaluator evaluator = new Evaluator(testDB, isProPost, "baseline");
evaluator.outputToFile();

evaluator = new Evaluator(testDB, isAntiPost, "baseline");
evaluator.outputToFile();


def comparator = new SimpleRankingComparator(testDB)
comparator.setBaseline(testTruth_postPro)

// Choosing what metrics to report
def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
double [] score = new double[metrics.size()]

try {
    for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(isProPost)
    }
    //Storing the performance values of the current fold

    System.out.println("\nArea under positive-class PR curve for isProPost: " + score[0])
    System.out.println("Area under negetive-class PR curve for isProPost: " + score[1])
    System.out.println("Area under ROC curve for isProPost: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}

comparator.setBaseline(testTruth_postAnti)

// Choosing what metrics to report
score = new double[metrics.size()]

try {
    for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(isAntiPost)
    }
    //Storing the performance values of the current fold

    System.out.println("\nArea under positive-class PR curve for isAntiPost: " + score[0])
    System.out.println("Area under negetive-class PR curve for isAntiPost: " + score[1])
    System.out.println("Area under ROC curve for isAntiPost: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}


testDB.close()
distributionDB.close()
truthDB.close()
dummy_test.close()
dummy_DB.close()
testTruth_postPro.close()
testTruth_postAnti.close()
testTruth_authPro.close()
testTruth_authAnti.close()