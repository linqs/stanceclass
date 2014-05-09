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

model.add predicate: "writesPost" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "participates" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "agreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "disagreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Author predicates for social attitudes e.g. sarcasm, nasty, attack
 */
model.add predicate: "sarcastic" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "nasty" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "attacks" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

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
 * Latent, open predicates for latent network
 */

model.add predicate: "supports" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "against" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Target predicates
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isAntiAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "isProPost" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isAntiPost" , types:[ArgumentType.UniqueID, ArgumentType.String]



model.add rule : (hasLabelPro(P, T)) >> isProPost(P, T) , weight : 1
model.add rule : (hasLabelAnti(P, T)) >> isAntiPost(P, T) , weight : 1

/*
 * Inserting data into the data store
 */
//fold = 1

//foldStr = "fold" + String.valueOf(fold) + java.io.File.separator;

Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition observed_te = new Partition(3);
Partition predict_te = new Partition(4);
Partition truth_te = new Partition(5);
Partition dummy_tr = new Partition(6);
Partition dummy_tr2 = new Partition(7);
Partition dummy_te = new Partition(8);
Partition dummy_te2 = new Partition(9);

//def dir = 'data'+java.io.File.separator+ foldStr + 'train'+java.io.File.separator;
def dir = 'data'+java.io.File.separator+ 'stance-dev'+java.io.File.separator + 'train'+java.io.File.separator;

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"prolabels.csv", ",");

inserter = data.getInserter(hasLabelAnti, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"antilabels.csv", ",");

inserter = data.getInserter(hasTopic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"post_topics.csv", ",");

inserter = data.getInserter(writesPost, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"author_posts.csv", ",");

inserter = data.getInserter(topic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"topics.csv", ",");

inserter = data.getInserter(participates, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"participates.csv", ",")

inserter = data.getInserter(agreesAuth, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authoragreement.csv",",");

inserter = data.getInserter(disagreesAuth, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authordisagreement.csv", ",");

inserter = data.getInserter(sarcastic, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"sarcasm.csv", ",");

inserter = data.getInserter(nasty, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"nastiness.csv", ",");

inserter = data.getInserter(attacks, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"attack.csv", ",");


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

/*
 * Used later on to populate training DB with all possible interactions
 */

inserter = data.getInserter(supports, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "interaction.csv", ",")

inserter = data.getInserter(against, dummy_tr2)
InserterUtils.loadDelimitedData(inserter, dir + "interaction.csv", ",")

/*db population */

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

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"antilabels.csv", ",");

inserter = data.getInserter(hasTopic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"post_topics.csv", ",");

inserter = data.getInserter(writesPost, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"author_posts.csv",",");

inserter = data.getInserter(topic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"topics.csv",",");

inserter = data.getInserter(participates, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"participates.csv",",")

inserter = data.getInserter(agreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authoragreement.csv",",");

inserter = data.getInserter(disagreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authordisagreement.csv", ",");

inserter = data.getInserter(sarcastic, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"sarcasm.csv", ",");

inserter = data.getInserter(nasty, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"nastiness.csv", ",");

inserter = data.getInserter(attacks, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"attack.csv", ",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(isProPost, truth_te)
InserterUtils.loadDelimitedData(inserter, testdir+"post_pro.csv",",");

inserter = data.getInserter(isProAuth, truth_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authorpro.csv", ",");

inserter = data.getInserter(isAntiPost, truth_te)
InserterUtils.loadDelimitedData(inserter, testdir+"post_anti.csv",",");

inserter = data.getInserter(isAntiAuth, truth_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authoranti.csv", ",");

/*supports and against*/

inserter = data.getInserter(supports, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "interaction.csv", ",")

inserter = data.getInserter(against, dummy_te2)
InserterUtils.loadDelimitedData(inserter, testdir + "interaction.csv", ",")

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

Database distributionDB = data.getDatabase(predict_tr, [sarcastic, nasty, attacks, agreesAuth, disagreesAuth, participates, hasLabelPro, hasTopic, writesPost, topic] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProPost, isProAuth, isAntiAuth, isAntiPost] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [supports, isProAuth, isAntiAuth, isProPost, isAntiPost] as Set)
Database dummy_DB2 = data.getDatabase(dummy_tr2, [against] as Set)

/* Populate isProPost in observed DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, isProPost);


/* Populate isProAuth in observed DB. */
//dbPop.populateFromDB(truthDB, isProAuth);

/* Populate isAntiPost in observed DB. */
dbPop.populateFromDB(dummy_DB, isAntiPost);


/* Populate isAntiAuth in observed DB. */
//dbPop.populateFromDB(truthDB, isAntiAuth);

dbPop.populateFromDB(dummy_DB, isProAuth);
dbPop.populateFromDB(dummy_DB, isAntiAuth);

/*
 * Populate distribution DB with all possible interactions
 */
dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, supports);

dbPop.populateFromDB(dummy_DB2, against);

println model;

Database testDB = data.getDatabase(predict_te, [sarcastic, nasty, attacks, agreesAuth, disagreesAuth, participates, hasLabelPro, hasTopic, writesPost, topic] as Set, observed_te);
Database testTruthDB = data.getDatabase(truth_te, [isProPost, isProAuth, isAntiAuth, isAntiPost] as Set)

Database dummy_test = data.getDatabase(dummy_te, [supports, isProAuth, isAntiAuth, isProPost, isAntiPost] as Set)
Database dummy_test2 = data.getDatabase(dummy_te2, [against] as Set)

/* Populate isProPost in test DB. */

DatabasePopulator test_pop = new DatabasePopulator(testDB);
test_pop.populateFromDB(testTruthDB, isProPost);


/* Populate isProAuth in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(dummy_test, isProAuth);
test_populator.populateFromDB(dummy_test, isProAuth);

test_populator.populateFromDB(dummy_test, isAntiPost);
test_populator.populateFromDB(dummy_test, isAntiAuth);

test_populator.populateFromDB(dummy_test, supports);
test_populator.populateFromDB(dummy_test2, against);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference()
System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

/* Evaluation */

def comparator = new DiscretePredictionComparator(testDB)
comparator.setBaseline(testTruthDB)
comparator.setResultFilter(new MaxValueFilter(isProPost, 1))
comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

Set<GroundAtom> groundings = Queries.getAllAtoms(testTruthDB, isProPost)
int totalTestExamples = groundings.size()
DiscretePredictionStatistics stats = comparator.compare(isProPost, totalTestExamples)
System.out.println("Accuracy: " + stats.getAccuracy())
System.out.println("F1: " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Precision: " + stats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Recall: " + stats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE))

comparator.setResultFilter(new MaxValueFilter(isProAuth, 1))
comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

Set<GroundAtom> authorGroundings = Queries.getAllAtoms(testTruthDB, isProAuth)
totalTestExamples = authorGroundings.size()
DiscretePredictionStatistics authorstats = comparator.compare(isProAuth, totalTestExamples)
System.out.println("Accuracy: " + authorstats.getAccuracy())
System.out.println("F1: " + authorstats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Precision: " + authorstats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Recall: " + authorstats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE))

testTruthDB.close()
testDB.close()
distributionDB.close()
truthDB.close()
