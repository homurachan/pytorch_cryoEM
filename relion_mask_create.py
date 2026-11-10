#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /***************************************************************************
#  *
#  * Author: "Sjors H.W. Scheres"
#  * MRC Laboratory of Molecular Biology
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * This complete copyright notice must be included in any revised version of the
#  * source code. Additional authorship citations may be added, but existing
#  * author citations must be preserved.
#  ***************************************************************************/
# Python adaptation, 2026. GPL-2.0-or-later. Based on the supplied RELION 3.1.2 SIRM v1p2.
# Standalone: no sibling modules, RELION executable, or WSL required.
# /***************************************************************************
#  *
#  * Authors:    Roberto Marabini	(roberto@cnb.csic.es)
#  *             Carlos Oscar S. Sorzano	(coss@cnb.csic.es)
#  *
#  * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307 USA
#  *
#  * All comments concerning this program package may be sent to the
#  * e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *  All comments concerning this program package may be sent to the
#  *  e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#  *              Sjors H.W. Scheres
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *  All comments concerning this program package may be sent to the
#  *  e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:		 J.R. Bilbao-Castro (jrbcast@ace.ual.es)
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *	All comments concerning this program package may be sent to the
#  *	e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
LABEL_TYPES = {'randomSeed': 'INT', 'rlnComment': 'STRING', 'rlnAreaId': 'INT', 'rlnAreaName': 'STRING', 'rlnBodyMaskName': 'STRING', 'rlnBodyKeepFixed': 'INT', 'rlnBodyReferenceName': 'STRING', 'rlnBodyRotateDirectionX': 'DOUBLE', 'rlnBodyRotateDirectionY': 'DOUBLE', 'rlnBodyRotateDirectionZ': 'DOUBLE', 'rlnBodyRotateRelativeTo': 'INT', 'rlnBodySigmaAngles': 'DOUBLE', 'rlnBodySigmaOffset': 'DOUBLE', 'rlnBodySigmaOffsetAngst': 'DOUBLE', 'rlnBodySigmaRot': 'DOUBLE', 'rlnBodySigmaTilt': 'DOUBLE', 'rlnBodySigmaPsi': 'DOUBLE', 'rlnBodyStarFile': 'STRING', 'rlnCtfAstigmatism': 'DOUBLE', 'rlnCtfBfactor': 'DOUBLE', 'rlnCtfMaxResolution': 'DOUBLE', 'rlnCtfValidationScore': 'DOUBLE', 'rlnCtfScalefactor': 'DOUBLE', 'rlnVoltage': 'DOUBLE', 'rlnDefocusU': 'DOUBLE', 'rlnDefocusV': 'DOUBLE', 'rlnDefocusAngle': 'DOUBLE', 'rlnSphericalAberration': 'DOUBLE', 'rlnChromaticAberration': 'DOUBLE', 'rlnDetectorPixelSize': 'DOUBLE', 'rlnCtfPowerSpectrum': 'STRING', 'rlnEnergyLoss': 'DOUBLE', 'rlnCtfFigureOfMerit': 'DOUBLE', 'rlnCtfImage': 'STRING', 'rlnLensStability': 'DOUBLE', 'rlnMagnification': 'DOUBLE', 'rlnPhaseShift': 'DOUBLE', 'rlnConvergenceCone': 'DOUBLE', 'rlnLongitudinalDisplacement': 'DOUBLE', 'rlnTransversalDisplacement': 'DOUBLE', 'rlnAmplitudeContrast': 'DOUBLE', 'rlnCtfValue': 'DOUBLE', 'rlnImageName': 'STRING', 'rlnImageOriginalName': 'STRING', 'rlnReconstructImageName': 'STRING', 'rlnImageId': 'INT', 'rlnEnabled': 'BOOL', 'rlnDataType': 'INT', 'rlnImageDimensionality': 'INT', 'rlnBeamTiltX': 'DOUBLE', 'rlnBeamTiltY': 'DOUBLE', 'rlnMtfFileName': 'STRING', 'rlnOpticsGroup': 'INT', 'rlnOpticsGroupName': 'STRING', 'rlnOddZernike': 'DOUBLE_VECTOR', 'rlnEvenZernike': 'DOUBLE_VECTOR', 'rlnImagePixelSize': 'DOUBLE', 'rlnMagMat00': 'DOUBLE', 'rlnMagMat01': 'DOUBLE', 'rlnMagMat10': 'DOUBLE', 'rlnMagMat11': 'DOUBLE', 'rlnCoordinateX': 'DOUBLE', 'rlnCoordinateY': 'DOUBLE', 'rlnCoordinateZ': 'DOUBLE', 'rlnMovieFrameNumber': 'INT', 'rlnNormCorrection': 'DOUBLE', 'rlnMagnificationCorrection': 'DOUBLE', 'rlnSamplingRate': 'DOUBLE', 'rlnSamplingRateX': 'DOUBLE', 'rlnSamplingRateY': 'DOUBLE', 'rlnSamplingRateZ': 'DOUBLE', 'rlnImageSize': 'INT', 'rlnImageSizeX': 'INT', 'rlnImageSizeY': 'INT', 'rlnImageSizeZ': 'INT', 'rlnMinimumValue': 'DOUBLE', 'rlnMaximumValue': 'DOUBLE', 'rlnAverageValue': 'DOUBLE', 'rlnStandardDeviationValue': 'DOUBLE', 'rlnSkewnessValue': 'DOUBLE', 'rlnKurtosisExcessValue': 'DOUBLE', 'rlnImageWeight': 'DOUBLE', 'rlnMaskName': 'STRING', 'rlnJobIsContinue': 'BOOL', 'rlnJobType': 'INT', 'rlnJobTypeName': 'STRING', 'rlnJoboptionType': 'INT', 'rlnJobOptionVariable': 'STRING', 'rlnJobOptionValue': 'STRING', 'rlnJobOptionGUILabel': 'STRING', 'rlnJobOptionDefaultValue': 'STRING', 'rlnJobOptionSliderMin': 'DOUBLE', 'rlnJobOptionSliderMax': 'DOUBLE', 'rlnJobOptionSliderStep': 'DOUBLE', 'rlnJobOptionHelpText': 'STRING', 'rlnJobOptionFilePattern': 'STRING', 'rlnJobOptionDirectoryDefault': 'STRING', 'rlnJobOptionMenuOptions': 'STRING', 'rlnMatrix_1_1': 'DOUBLE', 'rlnMatrix_1_2': 'DOUBLE', 'rlnMatrix_1_3': 'DOUBLE', 'rlnMatrix_2_1': 'DOUBLE', 'rlnMatrix_2_2': 'DOUBLE', 'rlnMatrix_2_3': 'DOUBLE', 'rlnMatrix_3_1': 'DOUBLE', 'rlnMatrix_3_2': 'DOUBLE', 'rlnMatrix_3_3': 'DOUBLE', 'rlnAccumMotionTotal': 'DOUBLE', 'rlnAccumMotionEarly': 'DOUBLE', 'rlnAccumMotionLate': 'DOUBLE', 'rlnMicrographId': 'INT', 'rlnMicrographName': 'STRING', 'rlnMicrographGainName': 'STRING', 'rlnMicrographDefectFile': 'STRING', 'rlnMicrographNameNoDW': 'STRING', 'rlnMicrographMovieName': 'STRING', 'rlnMicrographMetadata': 'STRING', 'rlnMicrographTiltAngle': 'DOUBLE', 'rlnMicrographTiltAxisDirection': 'DOUBLE', 'rlnMicrographTiltAxisOutOfPlane': 'DOUBLE', 'rlnMicrographOriginalPixelSize': 'DOUBLE', 'rlnMicrographPixelSize': 'DOUBLE', 'rlnMicrographPreExposure': 'DOUBLE', 'rlnMicrographDoseRate': 'DOUBLE', 'rlnMicrographBinning': 'DOUBLE', 'rlnMicrographFrameNumber': 'INT', 'rlnMotionModelVersion': 'INT', 'rlnMicrographStartFrame': 'INT', 'rlnMicrographEndFrame': 'INT', 'rlnMicrographShiftX': 'DOUBLE', 'rlnMicrographShiftY': 'DOUBLE', 'rlnMotionModelCoeffsIdx': 'INT', 'rlnMotionModelCoeff': 'DOUBLE', 'rlnEERUpsampling': 'INT', 'rlnEERGrouping': 'INT', 'rlnAccuracyRotations': 'DOUBLE', 'rlnAccuracyTranslations': 'DOUBLE', 'rlnAccuracyTranslationsAngst': 'DOUBLE', 'rlnAveragePmax': 'DOUBLE', 'rlnCurrentResolution': 'DOUBLE', 'rlnCurrentImageSize': 'INT', 'rlnSsnrMap': 'DOUBLE', 'rlnReferenceDimensionality': 'INT', 'rlnDataDimensionality': 'INT', 'rlnDiff2RandomHalves': 'DOUBLE', 'rlnEstimatedResolution': 'DOUBLE', 'rlnFourierCompleteness': 'DOUBLE', 'rlnOverallFourierCompleteness': 'DOUBLE', 'rlnGoldStandardFsc': 'DOUBLE', 'rlnGroupName': 'STRING', 'rlnGroupNumber': 'INT', 'rlnGroupNrParticles': 'INT', 'rlnGroupScaleCorrection': 'DOUBLE', 'rlnNrHelicalAsymUnits': 'INT', 'rlnHelicalTwist': 'DOUBLE', 'rlnHelicalTwistMin': 'DOUBLE', 'rlnHelicalTwistMax': 'DOUBLE', 'rlnHelicalTwistInitialStep': 'DOUBLE', 'rlnHelicalRise': 'DOUBLE', 'rlnHelicalRiseMin': 'DOUBLE', 'rlnHelicalRiseMax': 'DOUBLE', 'rlnHelicalRiseInitialStep': 'DOUBLE', 'rlnIsHelix': 'BOOL', 'rlnFourierSpaceInterpolator': 'INT', 'rlnLogLikelihood': 'DOUBLE', 'rlnMinRadiusNnInterpolation': 'INT', 'rlnNormCorrectionAverage': 'DOUBLE', 'rlnNrClasses': 'INT', 'rlnNrBodies': 'INT', 'rlnNrGroups': 'INT', 'rlnSpectralOrientabilityContribution': 'DOUBLE', 'rlnOriginalImageSize': 'INT', 'rlnPaddingFactor': 'DOUBLE', 'rlnClassDistribution': 'DOUBLE', 'rlnClassPriorOffsetX': 'DOUBLE', 'rlnClassPriorOffsetY': 'DOUBLE', 'rlnOrientationDistribution': 'DOUBLE', 'rlnPixelSize': 'DOUBLE', 'rlnReferenceSpectralPower': 'DOUBLE', 'rlnOrientationalPriorMode': 'INT', 'rlnReferenceImage': 'STRING', 'rlnSGDGradientImage': 'STRING', 'rlnSigmaOffsets': 'DOUBLE', 'rlnSigmaOffsetsAngst': 'DOUBLE', 'rlnSigma2Noise': 'DOUBLE', 'rlnReferenceSigma2': 'DOUBLE', 'rlnSigmaPriorRotAngle': 'DOUBLE', 'rlnSigmaPriorTiltAngle': 'DOUBLE', 'rlnSigmaPriorPsiAngle': 'DOUBLE', 'rlnSignalToNoiseRatio': 'DOUBLE', 'rlnTau2FudgeFactor': 'DOUBLE', 'rlnReferenceTau2': 'DOUBLE', 'rlnOverallAccuracyRotations': 'DOUBLE', 'rlnOverallAccuracyTranslations': 'DOUBLE', 'rlnOverallAccuracyTranslationsAngst': 'DOUBLE', 'rlnAdaptiveOversampleFraction': 'DOUBLE', 'rlnAdaptiveOversampleOrder': 'INT', 'rlnAutoLocalSearchesHealpixOrder': 'INT', 'rlnAvailableMemory': 'DOUBLE', 'rlnBestResolutionThusFar': 'DOUBLE', 'rlnCoarseImageSize': 'INT', 'rlnChangesOptimalOffsets': 'DOUBLE', 'rlnChangesOptimalOrientations': 'DOUBLE', 'rlnChangesOptimalClasses': 'DOUBLE', 'rlnCtfDataArePhaseFlipped': 'BOOL', 'rlnCtfDataAreCtfPremultiplied': 'BOOL', 'rlnExperimentalDataStarFile': 'STRING', 'rlnDoCorrectCtf': 'BOOL', 'rlnDoCorrectMagnification': 'BOOL', 'rlnDoCorrectNorm': 'BOOL', 'rlnDoCorrectScale': 'BOOL', 'rlnDoExternalReconstruct': 'BOOL', 'rlnDoRealignMovies': 'BOOL', 'rlnDoMapEstimation': 'BOOL', 'rlnDoStochasticGradientDescent': 'BOOL', 'rlnDoStochasticEM': 'BOOL', 'rlnExtReconsDataReal': 'STRING', 'rlnExtReconsDataImag': 'STRING', 'rlnExtReconsWeight': 'STRING', 'rlnExtReconsResult': 'STRING', 'rlnExtReconsResultStarfile': 'STRING', 'rlnDoFastSubsetOptimisation': 'BOOL', 'rlnSgdInitialIterations': 'INT', 'rlnSgdFinalIterations': 'INT', 'rlnSgdInBetweenIterations': 'INT', 'rlnSgdInitialResolution': 'DOUBLE', 'rlnSgdFinalResolution': 'DOUBLE', 'rlnSgdInitialSubsetSize': 'INT', 'rlnSgdFinalSubsetSize': 'INT', 'rlnSgdMuFactor': 'DOUBLE', 'rlnSgdSigma2FudgeInitial': 'DOUBLE', 'rlnSgdSigma2FudgeHalflife': 'INT', 'rlnSgdSkipAnneal': 'BOOL', 'rlnSgdSubsetSize': 'INT', 'rlnSgdWriteEverySubset': 'INT', 'rlnSgdMaxSubsets': 'INT', 'rlnSgdStepsize': 'DOUBLE', 'rlnDoAutoRefine': 'BOOL', 'rlnDoOnlyFlipCtfPhases': 'BOOL', 'rlnDoSolventFlattening': 'BOOL', 'rlnDoSolventFscCorrection': 'BOOL', 'rlnDoSkipAlign': 'BOOL', 'rlnDoSkipRotate': 'BOOL', 'rlnDoSplitRandomHalves': 'BOOL', 'rlnDoZeroMask': 'BOOL', 'rlnFixSigmaNoiseEstimates': 'BOOL', 'rlnFixSigmaOffsetEstimates': 'BOOL', 'rlnFixTauEstimates': 'BOOL', 'rlnHasConverged': 'BOOL', 'rlnHasHighFscAtResolLimit': 'BOOL', 'rlnHasLargeSizeIncreaseIterationsAgo': 'INT', 'rlnDoHelicalRefine': 'BOOL', 'rlnIgnoreHelicalSymmetry': 'BOOL', 'rlnFourierMask': 'STRING', 'rlnHelicalTwistInitial': 'DOUBLE', 'rlnHelicalRiseInitial': 'DOUBLE', 'rlnHelicalCentralProportion': 'DOUBLE', 'rlnNrHelicalNStart': 'INT', 'rlnHelicalMaskTubeInnerDiameter': 'DOUBLE', 'rlnHelicalMaskTubeOuterDiameter': 'DOUBLE', 'rlnHelicalSymmetryLocalRefinement': 'BOOL', 'rlnHelicalSigmaDistance': 'DOUBLE', 'rlnHelicalKeepTiltPriorFixed': 'BOOL', 'rlnLowresLimitExpectation': 'DOUBLE', 'rlnHighresLimitExpectation': 'DOUBLE', 'rlnHighresLimitSGD': 'DOUBLE', 'rlnDoIgnoreCtfUntilFirstPeak': 'BOOL', 'rlnIncrementImageSize': 'INT', 'rlnCurrentIteration': 'INT', 'rlnLocalSymmetryFile': 'STRING', 'rlnJoinHalvesUntilThisResolution': 'DOUBLE', 'rlnMagnificationSearchRange': 'DOUBLE', 'rlnMagnificationSearchStep': 'DOUBLE', 'rlnMaximumCoarseImageSize': 'INT', 'rlnMaxNumberOfPooledParticles': 'INT', 'rlnModelStarFile': 'STRING', 'rlnModelStarFile2': 'STRING', 'rlnNumberOfIterations': 'INT', 'rlnNumberOfIterWithoutResolutionGain': 'INT', 'rlnNumberOfIterWithoutChangingAssignments': 'INT', 'rlnOpticsStarFile': 'STRING', 'rlnOutputRootName': 'STRING', 'rlnParticleDiameter': 'DOUBLE', 'rlnRadiusMaskMap': 'INT', 'rlnRadiusMaskExpImages': 'INT', 'rlnRandomSeed': 'INT', 'rlnRefsAreCtfCorrected': 'BOOL', 'rlnSmallestChangesClasses': 'INT', 'rlnSmallestChangesOffsets': 'DOUBLE', 'rlnSmallestChangesOrientations': 'DOUBLE', 'rlnOrientSamplingStarFile': 'STRING', 'rlnSolventMaskName': 'STRING', 'rlnSolventMask2Name': 'STRING', 'rlnTauSpectrumName': 'STRING', 'rlnUseTooCoarseSampling': 'BOOL', 'rlnWidthMaskEdge': 'INT', 'rlnIsFlip': 'BOOL', 'rlnOrientationsID': 'INT', 'rlnOriginX': 'DOUBLE', 'rlnOriginY': 'DOUBLE', 'rlnOriginZ': 'DOUBLE', 'rlnOriginXPrior': 'DOUBLE', 'rlnOriginYPrior': 'DOUBLE', 'rlnOriginZPrior': 'DOUBLE', 'rlnOriginXAngst': 'DOUBLE', 'rlnOriginYAngst': 'DOUBLE', 'rlnOriginZAngst': 'DOUBLE', 'rlnOriginXPriorAngst': 'DOUBLE', 'rlnOriginYPriorAngst': 'DOUBLE', 'rlnOriginZPriorAngst': 'DOUBLE', 'rlnAngleRot': 'DOUBLE', 'rlnAngleRotPrior': 'DOUBLE', 'rlnAngleRotFlipRatio': 'DOUBLE', 'rlnAngleTilt': 'DOUBLE', 'rlnAngleTiltPrior': 'DOUBLE', 'rlnAnglePsi': 'DOUBLE', 'rlnAnglePsiPrior': 'DOUBLE', 'rlnAnglePsiFlipRatio': 'DOUBLE', 'rlnAnglePsiFlip': 'BOOL', 'rlnAngleRotNear': 'DOUBLE', 'rlnAngleTiltNear': 'DOUBLE', 'rlnAngleDeltaTheta': 'DOUBLE', 'rlnWeightVectorX': 'DOUBLE', 'rlnWeightVectorY': 'DOUBLE', 'rlnWeightVectorZ': 'DOUBLE', 'rlnWeightVectorValue': 'DOUBLE', 'rlnDeltaZ': 'DOUBLE', 'rlnParticleSerialNumber': 'INT', 'rlnMaskPickingRot': 'DOUBLE', 'rlnMaskPickingTilt': 'DOUBLE', 'rlnMaskPickingPSI': 'DOUBLE', 'rlnAutopickFigureOfMerit': 'DOUBLE', 'rlnHelicalTubeID': 'INT', 'rlnHelicalTubePitch': 'DOUBLE', 'rlnHelicalTrackLength': 'DOUBLE', 'rlnHelicalTrackLengthAngst': 'DOUBLE', 'rlnClassNumber': 'INT', 'rlnLogLikeliContribution': 'DOUBLE', 'rlnParticleId': 'INT', 'rlnParticleFigureOfMerit': 'DOUBLE', 'rlnKullbackLeiblerDivergence': 'DOUBLE', 'rlnRandomSubset': 'INT', 'rlnBeamTiltClass': 'INT', 'rlnParticleName': 'STRING', 'rlnOriginalParticleName': 'STRING', 'rlnNrOfSignificantSamples': 'INT', 'rlnNrOfFrames': 'INT', 'rlnAverageNrOfFrames': 'INT', 'rlnMovieFramesRunningAverage': 'INT', 'rlnMaxValueProbDistribution': 'DOUBLE', 'rlnParticleNumber': 'INT', 'rlnPipeLineJobCounter': 'INT', 'rlnPipeLineNodeType': 'INT', 'rlnPipeLineProcessType': 'INT', 'rlnPipeLineProcessStatus': 'INT', 'rlnFinalResolution': 'DOUBLE', 'rlnBfactorUsedForSharpening': 'DOUBLE', 'rlnParticleBoxFractionMolecularWeight': 'DOUBLE', 'rlnParticleBoxFractionSolventMask': 'DOUBLE', 'rlnFourierShellCorrelation': 'DOUBLE', 'rlnFourierShellCorrelationCorrected': 'DOUBLE', 'rlnFourierShellCorrelationParticleMolWeight': 'DOUBLE', 'rlnFourierShellCorrelationParticleMaskFraction': 'DOUBLE', 'rlnFourierShellCorrelationMaskedMaps': 'DOUBLE', 'rlnFourierShellCorrelationUnmaskedMaps': 'DOUBLE', 'rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps': 'DOUBLE', 'rlnAmplitudeCorrelationMaskedMaps': 'DOUBLE', 'rlnAmplitudeCorrelationUnmaskedMaps': 'DOUBLE', 'rlnDifferentialPhaseResidualMaskedMaps': 'DOUBLE', 'rlnDifferentialPhaseResidualUnmaskedMaps': 'DOUBLE', 'rlnFittedInterceptGuinierPlot': 'DOUBLE', 'rlnFittedSlopeGuinierPlot': 'DOUBLE', 'rlnCorrelationFitGuinierPlot': 'DOUBLE', 'rlnLogAmplitudesOriginal': 'DOUBLE', 'rlnLogAmplitudesMTFCorrected': 'DOUBLE', 'rlnLogAmplitudesWeighted': 'DOUBLE', 'rlnLogAmplitudesSharpened': 'DOUBLE', 'rlnLogAmplitudesIntercept': 'DOUBLE', 'rlnResolutionSquared': 'DOUBLE', 'rlnMolecularWeight': 'DOUBLE', 'rlnMtfValue': 'DOUBLE', 'rlnRandomiseFrom': 'DOUBLE', 'rlnUnfilteredMapHalf1': 'STRING', 'rlnUnfilteredMapHalf2': 'STRING', 'rlnIs3DSampling': 'BOOL', 'rlnIs3DTranslationalSampling': 'BOOL', 'rlnHealpixOrder': 'INT', 'rlnHealpixOrderOriginal': 'INT', 'rlnTiltAngleLimit': 'DOUBLE', 'rlnOffsetRange': 'DOUBLE', 'rlnOffsetStep': 'DOUBLE', 'rlnOffsetRangeOriginal': 'DOUBLE', 'rlnOffsetStepOriginal': 'DOUBLE', 'rlnHelicalOffsetStep': 'DOUBLE', 'rlnSamplingPerturbInstance': 'DOUBLE', 'rlnSamplingPerturbFactor': 'DOUBLE', 'rlnPsiStep': 'DOUBLE', 'rlnPsiStepOriginal': 'DOUBLE', 'rlnSymmetryGroup': 'STRING', 'rlnScheduleEdgeNumber': 'INT', 'rlnScheduleEdgeInputNodeName': 'STRING', 'rlnScheduleEdgeOutputNodeName': 'STRING', 'rlnScheduleEdgeIsFork': 'BOOL', 'rlnScheduleEdgeOutputNodeNameIfTrue': 'STRING', 'rlnScheduleEdgeBooleanVariable': 'STRING', 'rlnScheduleCurrentNodeName': 'STRING', 'rlnScheduleOriginalStartNodeName': 'STRING', 'rlnScheduleEmailAddress': 'STRING', 'rlnScheduleName': 'STRING', 'rlnScheduleJobName': 'STRING', 'rlnScheduleJobNameOriginal': 'STRING', 'rlnScheduleJobMode': 'STRING', 'rlnScheduleJobHasStarted': 'BOOL', 'rlnScheduleOperatorName': 'STRING', 'rlnScheduleOperatorType': 'STRING', 'rlnScheduleOperatorInput1': 'STRING', 'rlnScheduleOperatorInput2': 'STRING', 'rlnScheduleOperatorOutput': 'STRING', 'rlnScheduleBooleanVariableName': 'STRING', 'rlnScheduleBooleanVariableValue': 'BOOL', 'rlnScheduleBooleanVariableResetValue': 'BOOL', 'rlnScheduleFloatVariableName': 'STRING', 'rlnScheduleFloatVariableValue': 'DOUBLE', 'rlnScheduleFloatVariableResetValue': 'DOUBLE', 'rlnScheduleStringVariableName': 'STRING', 'rlnScheduleStringVariableValue': 'STRING', 'rlnScheduleStringVariableResetValue': 'STRING', 'rlnSelected': 'INT', 'rlnParticleSelectZScore': 'DOUBLE', 'rlnSortedIndex': 'INT', 'rlnStarFileMovieParticles': 'STRING', 'rlnPerFrameCumulativeWeight': 'DOUBLE', 'rlnPerFrameRelativeWeight': 'DOUBLE', 'rlnResolution': 'DOUBLE', 'rlnAngstromResolution': 'DOUBLE', 'rlnResolutionInversePixel': 'DOUBLE', 'rlnSpectralIndex': 'INT', 'rlnUnknownLabel': 'UNKNOWN'}
PROGRAM = 'mask_create'
OPTIONS = [('--i', 'str', '', 'Input map to use for thresholding to generate initial binary mask'), ('--o', 'str', 'mask.mrc', 'Output mask'), ('--and', 'str', '', 'Pixels in the initial mask will be one if the input AND this map are above the --ini_threshold value'), ('--or', 'str', '', 'Pixels in the initial mask will be one if the input OR this map are above the --ini_threshold value'), ('--and_not', 'str', '', 'Pixels in the initial mask will be one if the input is above the --ini_threshold AND this map is below it'), ('--or_not', 'str', '', 'Pixels in the initial mask will be one if the input is above the --ini_threshold OR this map is below it'), ('--ini_threshold', 'float', '0.01', 'Initial threshold for binarization'), ('--extend_inimask', 'float', '0', 'Extend initial binary mask this number of pixels'), ('--width_soft_edge', 'float', '0', 'Width (in pixels) of the additional soft edge on the binary mask'), ('--invert', 'bool', None, 'Invert the final mask'), ('--helix', 'bool', None, 'Generate a mask for 3D helix'), ('--lowpass', 'float', '-1', 'Lowpass filter (in Angstroms) for the input map, prior to binarization (default is none)'), ('--angpix', 'float', '-1', 'Pixel size (in Angstroms) for the lowpass filter'), ('--z_percentage', 'float', '0.3', 'This box length along the center of Z axis contains good information of the helix'), ('--j', 'int', '1', 'Compatibility option; Python does not start a RELION thread pool'), ('--denovo', 'bool', None, 'Create a mask de novo'), ('--box_size', 'int', '-1', 'The box size of the mask in pixels'), ('--inner_radius', 'float', '0', 'Inner radius of the masked region in pixels'), ('--outer_radius', 'float', '99999', 'Outer radius of the mask region in pixels'), ('--center_x', 'float', '0', 'X coordinate of the center of the mask in pixels'), ('--center_y', 'float', '0', 'Y coordinate of the center of the mask in pixels'), ('--center_z', 'float', '0', 'Z coordinate of the center of the mask in pixels')]
import argparse
import copy
import glob
import math
import os
from pathlib import Path
import random
import re
import sys

VERSION = '1.0.0'

def cli():
    p = argparse.ArgumentParser(description=f'Standalone Python {PROGRAM}, RELION 3.1.2 SIRM v1p2 port',
                                allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for flag, typ, default, helptext in OPTIONS:
        kw = dict(dest=flag[2:].replace('-','_'),help=helptext)
        if typ == 'bool': kw.update(action='store_true')
        else:
            convert = {'int':lambda x:int(float(x)), 'float':float, 'str':str}[typ]
            kw.update(type=convert)
            if default is None: kw.update(required=True)
            else: kw.update(default=convert(default))
        p.add_argument(flag,**kw)
    p.add_argument('--version',action='version',version=f'{PROGRAM} Python {VERSION}')
    p.add_argument('--data_root',default='',help='Root for relative paths referenced by input STAR files')
    if not any(x[0]=='--random_seed' for x in OPTIONS):
        p.add_argument('--random_seed',type=int,default=0,help='Python random seed; negative for random seed')
    return p.parse_args()

def fail(message):
    raise ValueError(message)

def warn(message):
    print('WARNING: '+message,file=sys.stderr)

def label(s):
    return s if s.startswith('_') else '_'+s

def fmt(x):
    return format(float(x),'.12g')

def suffix(path,tag):
    p=Path(path)
    return str(p.with_name(p.stem+tag+p.suffix))

class Token:
    def __init__(self,value,quoted=False): self.value,self.quoted=value,quoted

def star_tokens(text):
    """STAR lexer: quotes, comments, multiline semicolon fields, no backslash escaping."""
    lines=iter(text.splitlines())
    for line in lines:
        if line.startswith(';'):
            values=[line[1:]]
            for line in lines:
                if line.startswith(';'): break
                values.append(line)
            else: fail('Unterminated STAR semicolon field')
            yield Token('\n'.join(values),True)
            continue
        i=0
        while i<len(line):
            if line[i].isspace(): i+=1; continue
            if line[i]=='#': break
            if line[i] in "\"'":
                q=line[i]; start=i=i+1
                while i<len(line) and not (line[i]==q and (i+1==len(line) or line[i+1].isspace() or line[i+1]=='#')): i+=1
                if i==len(line): fail('Unterminated STAR quote: '+line)
                yield Token(line[start:i],True); i+=1
            else:
                start=i
                while i<len(line) and not line[i].isspace(): i+=1
                yield Token(line[start:i])

def star_special(t):
    return not t.quoted and (t.value.startswith('_') or t.value.lower().startswith(('data_','save_')) or t.value.lower() in ('loop_','stop_','global_'))

class Table:
    def __init__(self,labels=None,rows=None):
        self.labels=list(labels or []); self.rows=list(rows or [])
    def col(self,key):
        key=label(key)
        if key not in self.labels: fail('Missing STAR column '+key)
        return self.labels.index(key)
    def add(self,key,value='?'):
        key=label(key)
        if key not in self.labels:
            self.labels.append(key)
            for row in self.rows: row.append(str(value))
        return self.col(key)
    def get(self,row,key,default=None):
        if label(key) not in self.labels:
            if default is not None: return default
            fail('Missing STAR column '+label(key))
        return row[self.col(key)]

class Block:
    def __init__(self,name): self.name=name; self.items=[]

class Star:
    def __init__(self,blocks=None): self.blocks=list(blocks or [])
    @classmethod
    def read(cls,path):
        tokens=list(star_tokens(Path(path).read_text(encoding='utf-8-sig')))
        result=cls(); block=None; i=0
        while i<len(tokens):
            t=tokens[i]; v=t.value; lo=v.lower()
            if not t.quoted and lo.startswith('data_'):
                block=Block(v[5:]); result.blocks.append(block); i+=1
            elif block is None: fail('STAR content before data_ block')
            elif lo=='loop_' and not t.quoted:
                i+=1; labs=[]
                while i<len(tokens) and tokens[i].value.startswith('_') and not tokens[i].quoted:
                    labs.append(tokens[i].value); i+=1
                if not labs or len(labs)!=len(set(labs)): fail('Invalid/duplicate STAR loop labels')
                rows=[]; row=[]
                while i<len(tokens) and not (not row and star_special(tokens[i])):
                    if star_special(tokens[i]): fail('Incomplete STAR loop row before '+tokens[i].value)
                    row.append(tokens[i].value); i+=1
                    if len(row)==len(labs): rows.append(row); row=[]
                if row: fail('Incomplete STAR loop row')
                block.items.append(Table(labs,rows))
                if i<len(tokens) and tokens[i].value.lower()=='stop_': i+=1
            elif v.startswith('_') and not t.quoted:
                if i+1>=len(tokens) or star_special(tokens[i+1]): fail('Missing STAR scalar value '+v)
                block.items.append((v,tokens[i+1].value)); i+=2
            else: fail('Unsupported STAR token '+v)
        return result
    def tables(self):
        for b in self.blocks:
            for t in b.items:
                if isinstance(t,Table): yield b,t
    def table(self,name=None):
        pairs=list(self.tables())
        if name and name!='discover':
            matches=[(b,t) for b,t in pairs if b.name==name.removeprefix('data_')]
        else:
            matches=[(b,t) for b,t in pairs if b.name in ('particles','micrographs','movies')]
            if not matches: matches=[(b,t) for b,t in pairs if b.name!='optics']
        if not matches: fail('No matching STAR loop: '+str(name))
        return matches[0]
    def text(self):
        out=['# version 30001','']
        for b in self.blocks:
            out+=['data_'+b.name,'']
            for item in b.items:
                if isinstance(item,Table):
                    out+=['loop_']+[f'{x} #{i+1}' for i,x in enumerate(item.labels)]
                    for row in item.rows:
                        vals=[star_quote(str(v)) for v in row]
                        out.append('\n'.join(vals) if any('\n' in v for v in vals) else ' '.join(vals))
                else:
                    k,v=item; q=star_quote(str(v)); out.append(k+('\n' if '\n' in q else ' ')+q)
                out.append('')
        return '\n'.join(out)+'\n'
    def write(self,path):
        import tempfile
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.'+Path(path).name+'.',suffix='.tmp',dir=Path(path).resolve().parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream: stream.write(self.text())
            os.replace(tmp,path)
        finally:
            if Path(tmp).exists(): Path(tmp).unlink()

def star_quote(v):
    if '\n' in v or ('"' in v and "'" in v):
        if any(x.startswith(';') for x in v.splitlines()): fail('STAR multiline value contains a semicolon at line start')
        return ';'+v+'\n;'
    if not v or any(c.isspace() for c in v) or v[0] in '_#;\"\'' or v.lower().startswith(('data_','save_')) or v.lower() in ('loop_','stop_','global_'):
        q="'" if "'" not in v else '"'
        return q+v+q
    return v

def star_table(name,labels,rows):
    b=Block(name); b.items.append(Table([label(x) for x in labels],[[str(v) for v in r] for r in rows])); return Star([b])

def resolve_image(name,starpath=None,data_root=''):
    prefix=''
    if '@' in name:
        idx,name=name.split('@',1); prefix=idx+'@'
    p=Path(name)
    if p.is_absolute() and p.exists(): return prefix+str(p)
    candidates=[]
    if data_root: candidates.append(Path(data_root)/p)
    candidates.append(p)
    if starpath:
        # RELION paths are relative to project root, which can be a parent of the STAR.
        candidates.extend(parent/p for parent in Path(starpath).resolve().parents)
    for candidate in candidates:
        if candidate.exists(): return prefix+str(candidate.resolve())
    fail('Cannot resolve image '+name+'; use --data_root for the RELION project root')

def pixel_for(star,table,row,fallback=1):
    if '_rlnImagePixelSize' in table.labels: return float(table.get(row,'rlnImagePixelSize'))
    for b,t in star.tables():
        if b.name=='optics':
            gid=table.get(row,'rlnOpticsGroup')
            for r in t.rows:
                if t.get(r,'rlnOpticsGroup')==gid:
                    for key in ('rlnImagePixelSize','rlnMicrographPixelSize','rlnMicrographOriginalPixelSize'):
                        if label(key) in t.labels: return float(t.get(r,key))
                    return fallback
            fail('Particle refers to missing optics group '+gid)
    return fallback

def entrypoint():
    try: main(cli())
    except (ValueError,OSError,ImportError,IndexError,KeyError) as e:
        print('ERROR: '+str(e),file=sys.stderr); sys.exit(1)


# Heavy libraries are loaded only for operations that need them.
def numerical(scipy=False,mrc=True):
    global np, mrcfile, ndi
    import numpy as np
    if mrc: import mrcfile
    if scipy:
        from scipy import ndimage as ndi

def read_mrc(name):
    index=None
    if '@' in str(name):
        s,name=str(name).split('@',1); index=int(s)-1
        if index<0: fail('MRC stack indices are one based')
    with mrcfile.mmap(name,permissive=False,mode='r') as m:
        if tuple(int(m.header[k]) for k in ('mapc','mapr','maps'))!=(1,2,3):
            fail('Nonstandard MRC axis mapping; reorder axes before processing')
        ap=float(m.voxel_size.x)
        if ap<=0: ap=1.0
        a=m.data
        if index is not None:
            if a.ndim==2 and index==0: pass
            elif a.ndim<3 or index>=a.shape[0]: fail('MRC stack index out of range')
            else: a=a[index]
        meta={'angpix':ap,'origin':tuple(float(m.header.origin[k]) for k in ('x','y','z')),
              'is_stack':bool(m.is_image_stack() or str(name).lower().endswith('.mrcs'))}
        return np.array(a,dtype=np.float64),meta

def write_mrc(path,a,ap=1.0,meta=None,stack=False):
    if '@' in str(path):
        number,path=str(path).split('@',1); index=int(number)-1
        if index<0 or a.ndim!=2: fail('Indexed stack output requires a positive one-based index and a 2D image')
        if not Path(path).exists():
            if index!=0: fail('For a new indexed stack, start with 1@filename.mrcs')
            return write_mrc(path,a[None,...],ap,meta,True)
        with mrcfile.mmap(path,mode='r') as old:
            shape=old.data.shape; count=shape[0] if len(shape)==3 else 1
            if shape[-2:]!=a.shape: fail('Indexed output shape differs from existing stack')
            oldap=float(old.voxel_size.x)
            if oldap>0 and not math.isclose(oldap,ap,rel_tol=1e-6): fail('Indexed output pixel size differs from existing stack')
        if index<count:
            with mrcfile.mmap(path,mode='r+') as m:
                if m.data.ndim==2: m.data[:]=a
                else: m.data[index]=a
                m.update_header_stats()
            return
        if index!=count: fail('Indexed output may replace a frame or append the next frame; gaps are not allowed')
        import tempfile
        fd,tmp=tempfile.mkstemp(suffix='.mrcs',dir=Path(path).resolve().parent); os.close(fd)
        try:
            with mrcfile.mmap(path,mode='r') as old,mrcfile.new_mmap(tmp,shape=(count+1,*a.shape),mrc_mode=2,overwrite=True) as out:
                out.set_image_stack(); out.voxel_size=ap
                for i in range(count): out.data[i]=old.data[i] if old.data.ndim==3 else old.data
                out.data[count]=a; out.update_header_stats()
            os.replace(tmp,path)
        finally:
            if Path(tmp).exists(): Path(tmp).unlink()
        return
    if Path(path).suffix.lower() not in ('.mrc','.mrcs','.map'): fail('Output must be .mrc, .mrcs, .map or explicitly supported .png')
    if not np.isfinite(ap) or ap<=0: fail('Pixel size must be positive')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with mrcfile.new(path,overwrite=True) as m:
        m.set_data(np.asarray(a,dtype=np.float32))
        if stack and a.ndim==3: m.set_image_stack()
        elif a.ndim==3: m.set_volume()
        m.voxel_size=ap
        if meta: m.header.origin=meta.get('origin',(0.,0.,0.))
        m.update_header_stats()

def round_rel(x):
    return np.copysign(np.floor(np.abs(x)+0.5),x).astype(np.int64)

def grids(shape,fourier=False):
    vec=[]
    for i,n in enumerate(shape):
        if fourier:
            v=np.arange(n//2+1,dtype=float) if i==len(shape)-1 else np.rint(np.fft.fftfreq(n)*n)
            if i<len(shape)-1 and n%2==0: v[n//2]=n//2
        else: v=np.arange(n,dtype=float)-n//2
        sh=[1]*len(shape); sh[i]=len(v); vec.append(v.reshape(sh))
    return vec

def radius(shape,fourier=False):
    return np.sqrt(sum(g*g for g in grids(shape,fourier)))

def lowpass_weights(shape,ap,res,width=2,direction=''):
    if width<0 or res<=0 or ap<=0: fail('Invalid filter width, resolution or pixel size')
    n=shape[-1]; r=radius(shape,True)
    if direction:
        direction=direction.upper()
        if direction not in ('X','Y','Z') or ('XYZ'.index(direction)>=len(shape)): fail('Invalid filter direction')
        r=np.abs(grids(shape,True)[-1-'XYZ'.index(direction)])
    cutoff=int(math.floor(n*ap/res+0.5)); half=width//2
    lo=max(0,cutoff-half); hi=cutoff+half
    if hi<=lo: return (r<cutoff).astype(float)
    return 0.5+0.5*np.cos(np.pi*np.clip((r-lo)/(hi-lo),0,1))

def b_weights(shape,ap,b):
    r=radius(shape,True); s=r/(shape[-1]*ap)
    with np.errstate(over='raise'):
        return np.where(r<=shape[-1]/2,np.exp(-b*s*s/4),0)

def filter_map(a,ap,res,width=2,high=False,direction='',rng=None):
    oldshape=a.shape
    if not high and len(set(a.shape))!=1:
        if a.ndim!=2: fail('Filtering requires a cubic 3D map')
        a=pad_noise(a,rng or RelionRNG(0))
    w=lowpass_weights(a.shape,ap,res,width,direction)
    if high: w=1-w
    result=np.fft.irfftn(np.fft.rfftn(a)*w,s=a.shape)
    return window(result,oldshape) if result.shape!=oldshape else result

def shell_sum(values,shell,n):
    return np.bincount(shell.ravel(),weights=np.asarray(values).ravel(),minlength=n)[:n]

def fsc(a,b,selection=None):
    if a.shape!=b.shape: fail('FSC input shapes differ')
    f=np.fft.rfftn(a); g=np.fft.rfftn(b)
    return fsc_fourier(f,g,a.shape,selection)

def fsc_fourier(f,g,shape,selection=None):
    shell=round_rel(radius(shape,True)); n=shape[-1]//2+1
    w=1. if selection is None else selection
    num=shell_sum((f*g.conj()).real*w,shell,n)
    den=np.sqrt(shell_sum(np.abs(f)**2*w,shell,n)*shell_sum(np.abs(g)**2*w,shell,n))
    return np.divide(num,den,out=np.zeros(n),where=den>0)

def shell_mean(values,shape,n=None):
    shell=round_rel(radius(shape,True)); n=n or shape[-1]//2+1
    c=shell_sum(np.ones_like(values,dtype=float),shell,n)
    return shell_sum(values,shell,n)/np.maximum(c,1)

def radial_values(curve,shape,outside=0):
    s=round_rel(radius(shape,True)); return np.where(s<len(curve),curve[np.minimum(s,len(curve)-1)],outside)

def fsc_weights(curve,shape):
    c=np.asarray(curve).copy(); bad=np.flatnonzero(c<0.0001)
    if len(bad): c[bad[0]:]=0
    return radial_values(np.sqrt(2*np.maximum(c,0)/(1+np.maximum(c,0))),shape)

class RelionRNG:
    """Portable glibc rand() stream, matching the supplied Linux CPU executable."""
    def __init__(self,seed=0):
        import time
        seed=int(time.time()) if seed<0 else seed
        seed=(seed & 0xffffffff) or 1
        state=[seed]
        for i in range(1,31):
            prev=state[-1]; prev=prev if prev<2147483648 else prev-4294967296
            state.append((16807*prev)%2147483647)
        state+=state[:3]
        for i in range(34,344): state.append((state[i-31]+state[i-3])&0xffffffff)
        self.state=state[-31:]; self.pos=0; self.pending=np.empty(0,dtype=np.uint32); self.gaussian_spare=None
    def raw(self,count):
        prefix=self.pending[:count]; self.pending=self.pending[len(prefix):]
        count-=len(prefix)
        def stream():
            st=self.state; p=self.pos
            for _ in range(count):
                v=(st[p]+st[(p+28)%31])&0xffffffff; st[p]=v; p=(p+1)%31
                yield v>>1
        values=np.fromiter(stream(),dtype=np.uint32,count=count)
        self.pos=(self.pos+count)%31
        return np.concatenate((prefix,values)) if len(prefix) else values
    def uniform(self,lo,hi,size):
        count=int(np.prod(size)) if not isinstance(size,int) else size
        values=self.raw(count).astype(np.float32)
        # rnd_unif arguments and its result are float, even in double precision RELION.
        span=np.float32(hi)-np.float32(lo); denom=np.float32(np.float32(2147483647)/span)
        result=np.float32(lo)+values/denom
        return result.reshape(size).astype(float)
    def normal(self,mu,sigma,count):
        mu=np.float32(mu); sigma=np.float32(sigma)
        if sigma==0: return np.full(count,mu,dtype=float)
        out=np.empty(count,dtype=np.float32); pos=0
        if self.gaussian_spare is not None and count:
            out[0]=mu+sigma*self.gaussian_spare; pos=1; self.gaussian_spare=None
        while pos<count:
            pairs=min(65536,max(1,(count-pos+1)//2)); raw=self.raw(pairs*2).reshape(-1,2)
            u=np.float32(-1)+raw.astype(np.float32)/np.float32(2147483647)*np.float32(2)
            w=np.sum(u.astype(float)**2,axis=1).astype(np.float32); valid=np.flatnonzero((w>0)&(w<1))
            if not len(valid): continue
            take=min(len(valid),(count-pos+1)//2); last=valid[take-1]+1
            if last<len(raw): self.pending=np.concatenate((raw[last:].ravel(),self.pending))
            selected=valid[:take]; ww=w[selected].astype(float)
            mult=np.sqrt(-2*np.log(ww)/ww).astype(np.float32)
            values=(u[selected]*mult[:,None]).ravel(); needed=min(count-pos,len(values))
            out[pos:pos+needed]=mu+sigma*values[:needed]; pos+=needed
            if needed<len(values): self.gaussian_spare=values[-1]
        return out.astype(float)

def pad_noise(x,rng):
    n=max(x.shape); padded=window(x,n); present=window(np.ones(x.shape,dtype=bool),n)
    padded[~present]=rng.normal(x.mean(),x.std(ddof=0),int(np.count_nonzero(~present)))
    return padded

def random_phases(a,index,rng):
    f=np.fft.rfftn(a); selected=radius(a.shape,True)>=index
    phase=rng.uniform(0,2*np.pi,int(np.count_nonzero(selected)))
    # Same half-complex convention as RELION; inverse FFT makes a real map.
    f[selected]=np.abs(f[selected])*np.exp(1j*phase)
    return np.fft.irfftn(f,s=a.shape)

def window(a,shape):
    if isinstance(shape,int): shape=(shape,)*a.ndim
    if len(shape)!=a.ndim or min(shape)<1: fail('Invalid output box')
    out=np.zeros(shape,dtype=a.dtype); src=[]; dst=[]
    for old,new in zip(a.shape,shape):
        lo=max(-old//2 if old%2==0 else -(old//2),-(new//2))
        hi=min(old-old//2,new-new//2)
        src.append(slice(lo+old//2,hi+old//2)); dst.append(slice(lo+new//2,hi+new//2))
    out[tuple(dst)]=a[tuple(src)]; return out

def resize_fourier(a,n):
    if len(set(a.shape))!=1: fail('Fourier resizing requires a square or cubic box')
    old=a.shape[-1]
    if n==old: return a.copy()
    f=np.fft.rfftn(a)/a.size
    shape=(n,)*a.ndim; target=np.zeros(shape[:-1]+(n//2+1,),dtype=complex)
    coords=grids(shape,True)
    # RELION windowFourierTransform copies half-complex samples without Nyquist splitting.
    idx=[]; valid=np.ones(target.shape,dtype=bool)
    for k,(g,oldn) in enumerate(zip(coords,a.shape)):
        if k==a.ndim-1: valid &= g<=oldn//2; idx.append(np.minimum(g,oldn//2).astype(int))
        else:
            valid &= np.abs(g)<=oldn//2
            if n>oldn and oldn%2==0: valid &= g!=-oldn//2
            idx.append(np.mod(g,oldn).astype(int))
    if n>old: valid &= radius(shape,True)<=old//2
    target[valid]=np.broadcast_to(f[tuple(idx)],target.shape)[valid]
    return np.fft.irfftn(target,s=shape)*np.prod(shape)

def euler_matrix(rot,tilt,psi):
    a,b,g=np.deg2rad([rot,tilt,psi]); ca,sa=np.cos(a),np.sin(a); cb,sb=np.cos(b),np.sin(b); cg,sg=np.cos(g),np.sin(g)
    return np.array([[cg*cb*ca-sg*sa,cg*cb*sa+sg*ca,-cg*sb],[-sg*cb*ca-cg*sa,-sg*cb*sa+cg*ca,sg*sb],[sb*ca,sb*sa,cb]])

def matrix_euler(m):
    ab=np.hypot(m[0,2],m[1,2])
    if ab>1e-10:
        a=np.arctan2(m[2,1],m[2,0]); g=np.arctan2(m[1,2],-m[0,2]); b=np.arctan2(ab,m[2,2])
    else:
        a=0.; b=0. if m[2,2]>0 else np.pi
        g=np.arctan2(-m[1,0],m[0,0]) if b==0 else np.arctan2(m[1,0],-m[0,0])
    return np.rad2deg([a,b,g])

def symmetry_operators(sym):
    name=sym or 'C1'; sym=name.upper(); identity=np.eye(3); lh=np.diag([1.,1.,-1.])
    rot=[]; mirrors=[]; inversion=False
    match=re.fullmatch('([CDS])([1-9][0-9]*)([VH]?)',sym)
    if match:
        kind,num,tail=match.groups(); num=int(num)
        if kind=='S':
            if num%2: fail('S symmetry order must be even')
            num//=2; inversion=True
        rot=[(num,[0,0,1])]
        if kind=='D': rot.append((2,[1,0,0]))
        if tail: mirrors.append([0,0,1] if tail=='H' else [1,0,0] if kind=='D' else [0,1,0])
    elif sym in ('CI','CS'):
        inversion=sym=='CI'; mirrors=[] if inversion else [[0,0,1]]
    elif sym in ('T','TD','TH'):
        rot=[(3,[0,0,1]),(2,[0,.816496,.577350])]; inversion=sym=='TH'
        if sym=='TD': mirrors=[[1.4142136,2.4494897,0]]
    elif sym in ('O','OH'):
        rot=[(3,[.5773502]*3),(4,[0,0,1])]
        if sym=='OH': mirrors=[[0,1,1]]
    elif sym in ('I','IH','I1','I2','I3','I4','I1H','I2H','I3H','I4H'):
        conv=sym.rstrip('H'); conv='I2' if conv=='I' else conv
        axes={
            'I1':[(2,[1,0,0]),(5,[.85065080702670,0,-.5257311142635]),(3,[.9341723640,.3568220765,0])],
            'I2':[(2,[0,0,1]),(5,[.525731114,0,.850650807]),(3,[0,.356822076,.934172364])],
            'I3':[(2,[-.5257311143,0,.8506508070]),(5,[0,0,1]),(3,[-.4911234778630044,.3568220764705179,.7946544753759428])],
            'I4':[(2,[.5257311143,0,.8506508070]),(5,[.8944271932547096,0,.4472135909903704]),(3,[.4911234778630044,.3568220764705179,.7946544753759428])]}
        rot=axes[conv]
        if sym.endswith('H'): mirrors=[{'I1':[0,0,-1],'I2':[1,0,0],'I3':[.850650807,0,.525731114],'I4':[.850650807,0,-.525731114]}[conv]]
    elif Path(name).is_file():
        for line in Path(name).read_text().splitlines():
            words=line.split('#',1)[0].replace(',',' ').split()
            if not words: continue
            if words[0]=='rot_axis': rot.append((int(words[1]),list(map(float,words[2:5]))))
            elif words[0]=='mirror_plane': mirrors.append(list(map(float,words[1:4])))
            elif words[0]=='inversion': inversion=True
            else: fail('Unknown symmetry directive '+words[0])
    else: fail('Unsupported point group '+sym)
    ops=[]
    for fold,axis in rot:
        v=np.array(axis,float); v/=np.linalg.norm(v); k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        for j in range(1,fold):
            angle=np.deg2rad(360*j/fold); r=(identity+np.sin(angle)*k+(1-np.cos(angle))*(k@k)).T
            r[np.abs(r)<1e-6]=0; ops.append((identity.copy(),r))
    if inversion: ops.append((lh.copy(),-identity))
    for axis in mirrors:
        v=np.array(axis,float); v/=np.linalg.norm(v); ops.append((lh.copy(),identity-2*np.outer(v,v)))
    # Diagonal traversal is the upstream closure order, important for BBR row order.
    n=0
    while n<len(ops):
        for j in range(n,-1,-1):
            for i in range(n+1):
                if max(i,j)!=n: continue
                l=ops[i][0]@ops[j][0]; r=ops[i][1]@ops[j][1]
                if np.allclose(l,identity,atol=1e-5,rtol=0) and np.allclose(r,identity,atol=1e-5,rtol=0): continue
                if not any(np.allclose(l,ll,atol=1e-5,rtol=0) and np.allclose(r,rr,atol=1e-5,rtol=0) for ll,rr in ops):
                    l[np.abs(l)<1e-6]=0; r[np.abs(r)<1e-6]=0; ops.append((l,r))
                    if len(ops)>500: fail('Symmetry generators do not close to a supported finite group')
        n+=1
    return [(identity,identity)]+ops

def symmetry_matrices(sym): return [r for l,r in symmetry_operators(sym)]

def symmetrize(x,sym):
    mats=symmetry_matrices(sym); out=x.copy(); center=np.array(x.shape)//2
    for mat in mats[1:]:
        m=mat[::-1,::-1]
        for start in range(0,x.shape[0],16):
            stop=min(start+16,x.shape[0]); coords=np.indices((stop-start,*x.shape[1:]),dtype=float)
            coords[0]+=start; coords-=center[:,None,None,None]
            shape=coords.shape; src=(m@coords.reshape(3,-1)+center[:,None]).reshape(shape)
            for k,n in enumerate(x.shape):
                src[k]=np.where((src[k]>=-1e-6)&(src[k]<=n-1+1e-6),np.clip(src[k],0,n-1),src[k])
            out[start:stop]+=ndi.map_coordinates(x,src,order=1,mode='constant',cval=0,prefilter=False)
    return out/len(mats)

def automask(a,threshold,extend,width):
    if width<0: fail('Soft edge width cannot be negative')
    mask=a>=threshold
    if not mask.any(): return np.zeros(a.shape)
    if not mask.all() and extend:
        # Only in-box neighbors count in RELION, including at the box boundary.
        if extend>0: mask |= ndi.distance_transform_edt(~mask)<extend
        else:
            size=abs(math.ceil(extend)); rr=np.indices((2*size+1,)*a.ndim)-size
            structure=np.sum(rr*rr,axis=0)<extend*extend
            mask=ndi.binary_erosion(mask,structure=structure,border_value=1)
    if not mask.any(): return np.zeros(a.shape)
    if width>0 and not mask.all():
        d=ndi.distance_transform_edt(~mask); return 0.5+0.5*np.cos(np.pi*np.minimum(d/width,1))
    return mask.astype(float)

def spherical_mask(shape,inner,outer,width,center=None):
    g=grids(shape); center=center or [0]*len(shape)
    r=np.sqrt(sum((v-c)**2 for v,c in zip(g,center)))
    if width==0: return ((r>=inner)&(r<outer)).astype(float)
    v=np.ones(shape)
    v=np.where(r<inner,0.5-0.5*np.cos(np.pi*np.clip((r-inner+width)/width,0,1)),v)
    return np.where(r>=outer,0.5+0.5*np.cos(np.pi*np.clip((r-outer)/width,0,1)),v)

def curve_table(curves,n,ap,name='fsc'):
    rows=[]
    for i in range(len(next(iter(curves.values())))):
        res=n*ap/i if i else 999.
        rows.append([i,fmt(1/res),fmt(res)]+[fmt(c[i]) for c in curves.values()])
    return star_table(name,['rlnSpectralIndex','rlnResolution','rlnAngstromResolution']+list(curves),rows)


def main(a):
    numerical(bool(not a.denovo and (a.extend_inimask or a.width_soft_edge>0)))
    if a.j<1: fail('--j must be positive')
    if a.width_soft_edge<0: fail('--width_soft_edge must be nonnegative')
    if a.denovo:
        if a.box_size<1: fail('--denovo requires positive --box_size')
        if a.inner_radius<0 or a.outer_radius<a.inner_radius: fail('Invalid inner/outer radius')
        out=spherical_mask((a.box_size,)*3,a.inner_radius,a.outer_radius,a.width_soft_edge,[a.center_z,a.center_y,a.center_x])
        ap=a.angpix if a.angpix>0 else 1.; meta=None
    else:
        if not a.i: fail('Provide --i or --denovo')
        x,meta=read_mrc(a.i); ap=a.angpix if a.angpix>0 else meta['angpix']
        if meta['is_stack']: fail('Mask creation expects a map, not a particle stack')
        if not np.isfinite(x).all(): fail('Input contains NaN/Inf')
        if a.lowpass>0: x=filter_map(x,ap,a.lowpass)
        for op in ('and','or','and_not','or_not'):
            path=getattr(a,op)
            if path:
                y,_=read_mrc(path)
                if x.shape!=y.shape: fail('Boolean mask inputs have different shapes')
                p=x>a.ini_threshold; q=y<a.ini_threshold if op.endswith('_not') else y>a.ini_threshold
                x=np.where(p&q if op.startswith('and') else p|q,a.ini_threshold+1,a.ini_threshold-1)
                break
        out=automask(x,a.ini_threshold,a.extend_inimask,a.width_soft_edge)
        if a.helix:
            if out.ndim!=3 or out.shape[0]<5 or not .1<=a.z_percentage<=.9 or a.width_soft_edge<.001:
                fail('--helix requires 3D, Z>=5, 0.1<=z_percentage<=0.9, and positive soft edge')
            z=np.abs(grids(out.shape)[0]); extent=out.shape[0]*a.z_percentage/2
            out*=.5+.5*np.cos(np.pi*np.clip((z-extent)/a.width_soft_edge,0,1))
    if a.invert: out=1-out
    write_mrc(a.o,out,ap,meta)
    print('Written:',a.o)

if __name__=='__main__': entrypoint()
