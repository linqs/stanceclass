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
import edu.umd.cs.psl.application.learning.weight.em.DualEM

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

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator

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
import edu.ucsc.cs.utils.ResultWriter;

import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.database.ReadOnlyDatabase;


//dataSet = "fourforums"
dataSet = "stance-classification"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

squared = true
subDir = args[1]
fold = args[2]
def dir = 'data'+java.io.File.separator + subDir + java.io.File.separator + fold + java.io.File.separator + 'train' + java.io.File.separator;
def testdir = 'data'+java.io.File.separator + subDir + java.io.File.separator + fold + java.io.File.separator + 'test' + java.io.File.separator;

def toytrain = 'data'+java.io.File.separator + 'toy' + java.io.File.separator + fold + java.io.File.separator + 'train' + java.io.File.separator;
def toytest = 'data'+java.io.File.separator + 'toy' + java.io.File.separator + fold + java.io.File.separator + 'test' + java.io.File.separator;

initialWeight = 5

PSLModel model = new PSLModel(this, data)

/*
 * Author predicates of the form: predicate(authorID, authorID, topic) 
 * or (authorID, topic) 
 * or (authorID, postID)
 * Observed predicates
 */

model.add predicate: "participates" , types:[ArgumentType.UniqueID, ArgumentType.String]
//model.add predicate: "agreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
//model.add predicate: "disagreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Author predicates for social attitudes e.g. sarcasm, nasty, attack
 */
model.add predicate: "sarcastic" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "nasty" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "attacks" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "agrees" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "responds" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "hasLabelPro" , types:[ArgumentType.UniqueID, ArgumentType.String]

/*
 * Auxiliary topic predicate
 */
model.add predicate: "topic" , types:[ArgumentType.String]
model.add predicate: "ideology" , types:[ArgumentType.String]

/* Latent ideology predicate */
model.add predicate: "hasIdeologyA" , types:[ArgumentType.UniqueID]

/*
 * Target predicates
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]



/* simple stance rules*/

//model.add rule : (agrees(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> ~isProAuth(A1, T), weight : initialWeight, squared:squared
//model.add rule : (agrees(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> (isProAuth(A1, T)), weight :initialWeight, squared:squared

//model.add rule : (~agrees(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~isProAuth(A2, T)) >> isProAuth(A1, T), weight : initialWeight
//model.add rule : (~agrees(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & (isProAuth(A2, T))) >> ~(isProAuth(A1, T)), weight :initialWeight

//model.add rule : (sarcastic(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> ~isProAuth(A1, T), weight : initialWeight, squared:squared
//model.add rule : (sarcastic(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> (isProAuth(A1, T)), weight :initialWeight, squared:squared

//model.add rule : (~sarcastic(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> isProAuth(A1, T), weight : initialWeight
//model.add rule : (~sarcastic(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> ~(isProAuth(A1, T)), weight :initialWeight

//model.add rule : (nasty(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> ~isProAuth(A1, T), weight : initialWeight, squared:squared
//model.add rule : (nasty(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> (isProAuth(A1, T)), weight :initialWeight, squared:squared

//model.add rule : (~nasty(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> isProAuth(A1, T), weight : initialWeight
//model.add rule : (~nasty(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> ~(isProAuth(A1, T)), weight :initialWeight

//model.add rule : (attacks(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & isProAuth(A2, T)) >> ~isProAuth(A1, T), weight : initialWeight, squared:squared
//model.add rule : (attacks(A1, A2, T) & (A1-A2) & participates(A1, T) & participates(A2, T) & ~(isProAuth(A2, T))) >> (isProAuth(A1, T)), weight :initialWeight, squared:squared

//model.add rule : (agrees(A1, A2, T) & agrees(A2, A1, T) & (A1-A2) & hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
//model.add rule : (agrees(A1, A2, T)  & agrees(A2, A1, T) & (A1-A2) & hasIdeologyA(A2)) >> hasIdeologyA(A1), weight :initialWeight

//model.add rule : (agrees(A1, A2, T) & agrees(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight
//model.add rule : (agrees(A1, A2, T) &  agrees(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A2)) >> ~hasIdeologyA(A1), weight :initialWeight


//model.add rule : (sarcastic(A1, A2, T) & sarcastic(A2, A1, T) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight
//model.add rule : (sarcastic(A1, A2, T)  & sarcastic(A2, A1, T) & (A1-A2) & hasIdeologyA(A2)) >> ~hasIdeologyA(A1), weight :initialWeight

//model.add rule : (sarcastic(A1, A2, T) & sarcastic(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
//model.add rule : (sarcastic(A1, A2, T)  & sarcastic(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A2)) >> hasIdeologyA(A1), weight :initialWeight


//model.add rule : (nasty(A1, A2, T) & nasty(A2, A1, T) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight
//model.add rule : (nasty(A1, A2, T) & nasty(A2, A1, T) & (A1-A2) & hasIdeologyA(A2)) >> ~hasIdeologyA(A1), weight :initialWeight

//model.add rule : (nasty(A1, A2, T) & nasty(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
//model.add rule : (nasty(A1, A2, T)  & nasty(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A2)) >> hasIdeologyA(A1), weight :initialWeight


//model.add rule : (attacks(A1, A2, T) & attacks(A2, A1, T) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight
//model.add rule : (attacks(A1, A2, T)  & attacks(A2, A1, T) & (A1-A2) & hasIdeologyA(A2)) >> ~hasIdeologyA(A1), weight :initialWeight

//model.add rule : (attacks(A1, A2, T) & attacks(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
//model.add rule : (attacks(A1, A2, T)  & attacks(A2, A1, T) & (A1-A2) & ~hasIdeologyA(A2)) >> hasIdeologyA(A1), weight :initialWeight




//model.add rule : (responds(A1, A2, T) & (A1 - A2) & isProAuth(A2, T) & participates(A1, T)) >> ~isProAuth(A1, T), weight : initialWeight, squared:squared
//model.add rule : (responds(A1, A2, T) & (A1 - A2) & ~isProAuth(A2, T) & participates(A2, T) & participates(A1, T)) >> isProAuth(A1, T), weight : initialWeight, squared:squared



//model.add rule: (hasIdeologyA(A) & participates(A, T)) >> isProAuth(A, T), weight : initialWeight
//model.add rule: (hasLabelPro(A, T)) >> hasIdeologyA(A) , weight : initialWeight
//model.add rule: (~hasIdeologyA(A) & participates(A, T)) >> ~isProAuth(A, T), weight : initialWeight
//model.add rule: (~hasLabelPro(A, T) & participates(A, T)) >> ~hasIdeologyA(A) , weight : initialWeight


model.add rule: (hasIdeologyA(A) & participates(A, "abortion")) >> isProAuth(A,"abortion"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "evolution")) >> isProAuth(A, "evolution"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "gaymarriage")) >> isProAuth(A, "gaymarriage"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "guncontrol")) >> isProAuth(A, "guncontrol"), weight : initialWeight, squared:squared

model.add rule: (hasIdeologyA(A) & participates(A, "abortion")) >> ~isProAuth(A, "abortion"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "evolution")) >> ~isProAuth(A, "evolution"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "gaymarriage")) >> ~isProAuth(A, "gaymarriage"), weight : initialWeight, squared:squared
model.add rule: (hasIdeologyA(A) & participates(A, "guncontrol")) >> ~isProAuth(A, "guncontrol"), weight : initialWeight, squared:squared

model.add rule: (~hasIdeologyA(A) & participates(A, "abortion")) >> ~isProAuth(A, "abortion"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "evolution")) >> ~isProAuth(A, "evolution"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "gaymarriage")) >> ~isProAuth(A, "gaymarriage"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "guncontrol")) >> ~isProAuth(A, "guncontrol"), weight : initialWeight, squared:squared

model.add rule: (~hasIdeologyA(A) & participates(A, "abortion")) >> isProAuth(A, "abortion"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "evolution")) >> isProAuth(A, "evolution"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "gaymarriage")) >> isProAuth(A, "gaymarriage"), weight : initialWeight, squared:squared
model.add rule: (~hasIdeologyA(A) & participates(A, "guncontrol")) >> isProAuth(A, "guncontrol"), weight : initialWeight, squared:squared

model.add rule: (isProAuth(A, "abortion")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "evolution")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "gaymarriage")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "guncontrol")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared

model.add rule: (~isProAuth(A, "abortion") & participates(A, "abortion")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "evolution") & participates(A, "evolution")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "gaymarriage") & participates(A, "gaymarriage")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "guncontrol") & participates(A, "guncontrol")) >> hasIdeologyA(A) , weight : initialWeight, squared:squared

model.add rule: (isProAuth(A, "abortion")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "evolution")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "gaymarriage")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (isProAuth(A, "guncontrol")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared

model.add rule: (~isProAuth(A, "abortion") & participates(A, "abortion")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "evolution") & participates(A, "evolution")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "gaymarriage") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared
model.add rule: (~isProAuth(A, "guncontrol") & participates(A, "guncontrol")) >> ~hasIdeologyA(A) , weight : initialWeight, squared:squared


/*
model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "evolution")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & isProAuth(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & isProAuth(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "gaymarriage") & isProAuth(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight

model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "evolution")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & isProAuth(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & isProAuth(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & isProAuth(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "gaymarriage") & isProAuth(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight

model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & ~isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & ~isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "gaymarriage") & ~isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight

model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & ~isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & ~isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & ~isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "gaymarriage") & ~isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight

model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & ~isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & ~isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "gaymarriage") & ~isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight

model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "abortion") & ~isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & ~isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "evolution") & ~isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (isProAuth(A, "gaymarriage") & ~isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight

model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "gaymarriage") & isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> hasIdeologyA(A), weight : initialWeight

model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "evolution") &  participates(A, "abortion") & participates(A, "evolution")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "gaymarriage") & participates(A, "abortion") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "abortion") & isProAuth(A, "guncontrol") & participates(A, "abortion") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & isProAuth(A, "gaymarriage") & participates(A, "evolution") & participates(A, "gaymarriage")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "evolution") & isProAuth(A, "guncontrol") & participates(A, "evolution") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight
model.add rule : (~isProAuth(A, "gaymarriage") & isProAuth(A, "guncontrol") & participates(A, "gaymarriage") & participates(A, "guncontrol")) >> ~hasIdeologyA(A), weight : initialWeight

*/

//Prior that the label given by the text classifier is indeed the stance label

//model.add rule : (hasLabelPro(A, T)) >> isProAuth(A, T) , weight : initialWeight, squared:squared
//model.add rule : (~(hasLabelPro(A, T))) >> ~isProAuth(A, T) , weight : initialWeight, squared:squared

/*
 * Inserting data into the data store
 */

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

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"hasLabelPro.csv", ",");

inserter = data.getInserter(topic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"topic.csv", ",");

/* uncomment to use real data not toy data*/
//inserter = data.getInserter(participates, observed_tr)
//InserterUtils.loadDelimitedData(inserter, dir+"participates.csv", ",")


inserter = data.getInserter(participates, observed_tr)
InserterUtils.loadDelimitedData(inserter, toytrain+"participates.csv", ",")

/*load sentiment predicates with soft truth values*/

inserter = data.getInserter(agrees, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"agreement_author.csv",",");

inserter = data.getInserter(sarcastic, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"sarcasm_author.csv", ",");

inserter = data.getInserter(nasty, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"nastiness_author.csv", ",");

inserter = data.getInserter(attacks, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"attack_author.csv", ",");

inserter = data.getInserter(responds, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir + "responds.csv", ",");

//inserter = data.getInserter(hasIdeologyA, observed_tr)
//InserterUtils.loadDelimitedDataTruth(inserter, dir+"hasIdeologyA_seed.csv", ",");


/*
 * Ground truth for training data for weight learning
 */
/* uncomment to use real data not toy data*/
//inserter = data.getInserter(isProAuth, truth_tr)
//InserterUtils.loadDelimitedDataTruth(inserter, dir+"isProAuth.csv", ",");


inserter = data.getInserter(isProAuth, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, toytrain+"toy.csv", ",");


/*
 * Used later on to populate training DB with all possible interactions
 */

inserter = data.getInserter(hasIdeologyA, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "hasIdeologyA.csv", ",")


/*db population for all possible stance atoms*/

/* uncomment to use real data not toy data*/
//inserter = data.getInserter(isProAuth, dummy_tr)
//InserterUtils.loadDelimitedDataTruth(inserter, dir + "isProAuth.csv", ",")


inserter = data.getInserter(isProAuth, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, toytrain + "toy.csv", ",")

/*
 * Testing split for model inference
 * Observed partitions
 */

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"hasLabelPro.csv", ",");

inserter = data.getInserter(topic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"topic.csv",",");

/* uncomment to use real data not toy data*/
//inserter = data.getInserter(participates, observed_te)
//InserterUtils.loadDelimitedData(inserter, testdir+"participates.csv",",");


inserter = data.getInserter(participates, observed_te)
InserterUtils.loadDelimitedData(inserter, toytest+"participates.csv",",");



inserter = data.getInserter(agrees, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"agreement_author.csv",",");

inserter = data.getInserter(sarcastic, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"sarcasm_author.csv", ",");

inserter = data.getInserter(nasty, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"nastiness_author.csv", ",");

inserter = data.getInserter(attacks, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"attack_author.csv", ",");

inserter = data.getInserter(responds, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir + "responds.csv", ",");


//inserter = data.getInserter(hasIdeologyA, observed_te)
//InserterUtils.loadDelimitedDataTruth(inserter, testdir+"hasIdeologyA_seed.csv", ",");

/*for toy data experiments*/
inserter = data.getInserter(isProAuth, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, toytest + "toy_obs.csv", ",");


/*
 * Random variable partitions
 */
/* uncomment to use real data not toy data*/
//inserter = data.getInserter(isProAuth, authProTruth)
//InserterUtils.loadDelimitedDataTruth(inserter, testdir+"isProAuth.csv", ",");


inserter = data.getInserter(isProAuth, authProTruth)
InserterUtils.loadDelimitedDataTruth(inserter, toytest+"toy.csv", ",");

/*supports and against*/

inserter = data.getInserter(hasIdeologyA, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "hasIdeologyA.csv", ",")


/*to populate testDB with the correct rvs */
/* uncomment to use real data not toy data*/
//inserter = data.getInserter(isProAuth, dummy_te)
//InserterUtils.loadDelimitedDataTruth(inserter, testdir + "isProAuth.csv", ",")

inserter = data.getInserter(isProAuth, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, toytest + "toy.csv", ",")

/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [responds, hasLabelPro, sarcastic, nasty, attacks, agrees, participates, topic] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProAuth] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [hasIdeologyA, isProAuth] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, isProAuth);

/*
 * Populate distribution DB with all possible interactions
 */
dbPop.populateFromDB(dummy_DB, hasIdeologyA);


DualEM weightLearning = new DualEM(model, distributionDB, truthDB, cb);
weightLearning.learn();
weightLearning.close();

println model;

Database testDB = data.getDatabase(predict_te, [responds, hasLabelPro, sarcastic, nasty, attacks, agrees, participates, topic] as Set, observed_te);
Database testTruth_authPro = data.getDatabase(authProTruth, [isProAuth] as Set)

Database dummy_test = data.getDatabase(dummy_te, [hasIdeologyA, isProAuth] as Set)

/* Populate in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(dummy_test, isProAuth);
test_populator.populateFromDB(dummy_test, hasIdeologyA);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference();

/*output prediction results */
Evaluator evaluator = new Evaluator(testDB, isProAuth, "psl_authorstance_topicIdeology", fold);
evaluator.outputToFile();

///*output prediction results */
//evaluator = new Evaluator(testDB, supports, "supports", fold);
//evaluator.outputToFile();

/*output prediction results */
evaluator = new Evaluator(testDB, hasIdeologyA, "ideologyA_authorstance_topics", fold);
evaluator.outputToFile();

/* Accuracy */
def discComp = new DiscretePredictionComparator(testDB)
discComp.setBaseline(testTruth_authPro)
discComp.setResultFilter(new MaxValueFilter(isProAuth, 1))
discComp.setThreshold(0.5) // treat best value as true as long as it is nonzero

Set<GroundAtom> groundings = Queries.getAllAtoms(testTruth_authPro, isProAuth)
int totalTestExamples = groundings.size()
DiscretePredictionStatistics stats = discComp.compare(isProAuth, totalTestExamples)
System.out.println("Accuracy: " + stats.getAccuracy())
accuracy = (double) stats.getAccuracy()

def comparator = new SimpleRankingComparator(testDB)
comparator.setBaseline(testTruth_authPro)

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
    
    ResultWriter rs = new ResultWriter(score, fold, 'result_topicIdeology.txt')
    rs.write()
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}

testDB.close()
distributionDB.close()
truthDB.close()
dummy_test.close()
dummy_DB.close()
testTruth_authPro.close()