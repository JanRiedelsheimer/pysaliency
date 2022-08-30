from __future__ import absolute_import, print_function, division

import pytest
from pytest import approx

import unittest
import numpy as np
from scipy.stats import kurtosis, skew

import pysaliency
from pysaliency.datasets import remove_out_of_stimulus_fixations
import pysaliency.external_datasets
from pysaliency.utils import remove_trailing_nans


def _location(location):
    if location is not None:
        return str(location)
    return location


def entropy(labels):
    counts = np.bincount(labels)
    weights = counts / np.sum(counts)
    return -np.sum(weights*np.log(weights)) / np.log(2)


@pytest.mark.slow
@pytest.mark.download
def test_toronto(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_toronto(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('toronto/stimuli.hdf5').check()
        assert location.join('toronto/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 120
    for n in range(len(stimuli.stimuli)):
        assert stimuli.shapes[n] == (511, 681, 3)
        assert stimuli.sizes[n] == (511, 681)

    assert len(fixations.x) == 11199

    assert np.mean(fixations.x) == approx(345.7466738101616)
    assert np.mean(fixations.y) == approx(244.11393874453077)
    assert np.mean(fixations.t) == approx(0.0)
    assert np.mean(fixations.lengths) == approx(0.0)

    assert np.std(fixations.x) == approx(132.7479359296397)
    assert np.std(fixations.y) == approx(82.89667109045186)
    assert np.std(fixations.t) == approx(0.0)
    assert np.std(fixations.lengths) == approx(0.0)

    assert kurtosis(fixations.x) == approx(-0.40985986581959066)
    assert kurtosis(fixations.y) == approx(0.2748036777667475)
    assert kurtosis(fixations.t) == approx(-3.0)
    assert kurtosis(fixations.lengths) == approx(-3.0)

    assert skew(fixations.x) == approx(-0.09509166105451604)
    assert skew(fixations.y) == approx(-0.08674038899319877)
    assert skew(fixations.t) == approx(0.0)
    assert skew(fixations.lengths) == approx(0.0)

    assert entropy(fixations.n) == approx(6.8939709237615405)
    assert (fixations.n == 0).sum() == 130

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.matlab
@pytest.mark.skip_octave
def test_cat2000_train(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_cat2000_train(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('CAT2000_train/stimuli.hdf5').check()
        assert location.join('CAT2000_train/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 2000
    assert set(stimuli.sizes) == {(1080, 1920)}
    assert set(stimuli.attributes.keys()) == {'category'}
    assert np.all(np.array(stimuli.attributes['category'][0:100]) == 0)
    assert np.all(np.array(stimuli.attributes['category'][100:200]) == 1)

    assert len(fixations.x) == 672053

    assert np.mean(fixations.x) == approx(977.9570036886972)
    assert np.mean(fixations.y) == approx(536.8014098590438)
    assert np.mean(fixations.t) == approx(10.95831429961625)
    assert np.mean(fixations.lengths) == approx(9.95831429961625)

    assert np.std(fixations.x) == approx(265.521305397389)
    assert np.std(fixations.y) == approx(200.3874894751514)
    assert np.std(fixations.t) == approx(6.881491455270027)
    assert np.std(fixations.lengths) == approx(6.881491455270027)

    assert kurtosis(fixations.x) == approx(0.8377433175079028)
    assert kurtosis(fixations.y) == approx(0.15890436764279947)
    assert kurtosis(fixations.t) == approx(0.08351046096368542)
    assert kurtosis(fixations.lengths) == approx(0.08351046096368542)

    assert skew(fixations.x) == approx(0.07428576098144545)
    assert skew(fixations.y) == approx(0.27425191693049106)
    assert skew(fixations.t) == approx(0.5874222148956657)
    assert skew(fixations.lengths) == approx(0.5874222148956657)

    assert entropy(fixations.n) == approx(10.955266908462857)
    assert (fixations.n == 0).sum() == 307

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
def test_cat2000_test(location):
    real_location = _location(location)

    stimuli = pysaliency.external_datasets.get_cat2000_test(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('CAT2000_test/stimuli.hdf5').check()

    assert len(stimuli.stimuli) == 2000
    assert set(stimuli.sizes) == {(1080, 1920)}
    assert set(stimuli.attributes.keys()) == {'category'}
    assert np.all(np.array(stimuli.attributes['category'][0:100]) == 0)
    assert np.all(np.array(stimuli.attributes['category'][100:200]) == 1)


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
@pytest.mark.matlab
def test_mit1003(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_mit1003(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('MIT1003/stimuli.hdf5').check()
        assert location.join('MIT1003/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 1003
    for n in range(len(stimuli.stimuli)):
        assert max(stimuli.sizes[n]) == 1024

    assert len(fixations.x) == 104171

    assert np.mean(fixations.x) == approx(487.13683496521253)
    assert np.mean(fixations.y) == approx(392.72728829760155)
    assert np.mean(fixations.t) == approx(1.5039892740461995)
    assert np.mean(fixations.lengths) == approx(3.3973754691804823)

    assert np.std(fixations.x) == approx(190.0203102093757)
    assert np.std(fixations.y) == approx(159.99210430350126)
    assert np.std(fixations.t) == approx(0.816414737693668)
    assert np.std(fixations.lengths) == approx(2.5433689996843354)

    assert kurtosis(fixations.x) == approx(-0.39272472247196033)
    assert kurtosis(fixations.y) == approx(0.6983793465837596)
    assert kurtosis(fixations.t) == approx(-1.2178525798721818)
    assert kurtosis(fixations.lengths) == approx(-0.45897225172578704)

    assert skew(fixations.x) == approx(0.2204976032609953)
    assert skew(fixations.y) == approx(0.6445191904777621)
    assert skew(fixations.t) == approx(0.08125182887100482)
    assert skew(fixations.lengths) == approx(0.5047182860999948)

    assert entropy(fixations.n) == approx(9.954348058662386)
    assert (fixations.n == 0).sum() == 121

    assert 'duration_hist' in fixations.__attributes__
    assert 'duration' in fixations.__attributes__
    assert len(fixations.duration_hist) == len(fixations.x)
    assert len(fixations.duration) == len(fixations.x)
    for i in range(len(fixations.x)):
        assert len(remove_trailing_nans(fixations.duration_hist[i])) == len(remove_trailing_nans(fixations.x_hist[i]))

    assert 'durations' in fixations.scanpath_fixation_attributes
    assert len(fixations.scanpath_fixation_attributes['durations']) == len(fixations.train_xs)
    for i in range(len(fixations.train_xs)):
        assert len(remove_trailing_nans(fixations.scanpath_fixation_attributes['durations'][i])) == len(remove_trailing_nans(fixations.train_xs[i]))

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))

@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
@pytest.mark.matlab
def test_mit1003_onesize(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_mit1003_onesize(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('MIT1003_onesize/stimuli.hdf5').check()
        assert location.join('MIT1003_onesize/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 463
    for n in range(len(stimuli.stimuli)):
        assert stimuli.sizes[n] == (768, 1024)

    assert len(fixations.x) == 48771
    assert (fixations.n == 0).sum() == 121


if __name__ == '__main__':
    unittest.main()


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_stimuli(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    stimuli_train, stimuli_val, stimuli_test = pysaliency.external_datasets.salicon._get_SALICON_stimuli(location=real_location, name='SALICONfoobar')

    assert isinstance(stimuli_train, pysaliency.FileStimuli)
    assert isinstance(stimuli_val, pysaliency.FileStimuli)
    assert isinstance(stimuli_test, pysaliency.FileStimuli)
    assert location.join('SALICONfoobar/stimuli_train.hdf5').check()
    assert location.join('SALICONfoobar/stimuli_val.hdf5').check()
    assert location.join('SALICONfoobar/stimuli_test.hdf5').check()

    assert len(stimuli_train) == 10000
    assert len(stimuli_val) == 5000
    assert len(stimuli_test) == 5000

    assert set(stimuli_train.sizes) == set([(480, 640)])
    assert set(stimuli_val.sizes) == set([(480, 640)])
    assert set(stimuli_test.sizes) == set([(480, 640)])


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2015_mouse(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2015', fixation_type='mouse')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()

    assert len(fixations_train.x) == 68992355

    assert np.mean(fixations_train.x) == approx(313.0925573565361)
    assert np.mean(fixations_train.y) == approx(229.669921428251)
    assert np.mean(fixations_train.t) == approx(2453.3845915246698)
    assert np.mean(fixations_train.lengths) == approx(0.0)

    assert np.std(fixations_train.x) == approx(147.69997888974905)
    assert np.std(fixations_train.y) == approx(96.52066518492143)
    assert np.std(fixations_train.t) == approx(1538.7280458609941)
    assert np.std(fixations_train.lengths) == approx(0.0)

    assert kurtosis(fixations_train.x) == approx(-0.8543758617424033)
    assert kurtosis(fixations_train.y) == approx(-0.6277250557240337)
    assert kurtosis(fixations_train.t) == approx(19515.32829536525)
    assert kurtosis(fixations_train.lengths) == approx(-3.0)

    assert skew(fixations_train.x) == approx(0.08274147964197842)
    assert skew(fixations_train.y) == approx(0.10465863071610296)
    assert skew(fixations_train.t) == approx(55.69180106087239)
    assert skew(fixations_train.lengths) == approx(0.0)

    assert entropy(fixations_train.n) == approx(13.278169650429593)
    assert (fixations_train.n == 0).sum() == 6928


    assert len(fixations_val.x) == 38846998

    assert np.mean(fixations_val.x) == approx(311.44141923141655)
    assert np.mean(fixations_val.y) == approx(229.10522205602607)
    assert np.mean(fixations_val.t) == approx(2463.950701930687)
    assert np.mean(fixations_val.lengths) == approx(0.0)

    assert np.std(fixations_val.x) == approx(149.34417260369818)
    assert np.std(fixations_val.y) == approx(97.93170200208576)
    assert np.std(fixations_val.t) == approx(1408.3339394913962)
    assert np.std(fixations_val.lengths) == approx(0.0)

    assert kurtosis(fixations_val.x) == approx(-0.8449322083004356)
    assert kurtosis(fixations_val.y) == approx(-0.6136372253463405)
    assert kurtosis(fixations_val.t) == approx(-1.1157482867740718)
    assert kurtosis(fixations_val.lengths) == approx(-3.0)

    assert skew(fixations_val.x) == approx(0.08926920530231194)
    assert skew(fixations_val.y) == approx(0.10168032060729842)
    assert skew(fixations_val.t) == approx(0.05444269756551158)
    assert skew(fixations_val.lengths) == approx(0.0)

    assert entropy(fixations_val.n) == approx(12.279414832007888)
    assert (fixations_val.n == 0).sum() == 8244

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2015_fixations(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2015', fixation_type='fixations')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()

    assert len(fixations_train.x) == 3171533

    assert np.mean(fixations_train.x) == approx(310.93839540689)
    assert np.mean(fixations_train.y) == approx(217.7589979356986)
    assert np.mean(fixations_train.t) == approx(5.020693147446361)
    assert np.mean(fixations_train.lengths) == approx(0.0)

    assert np.std(fixations_train.x) == approx(131.0672366442846)
    assert np.std(fixations_train.y) == approx(86.33526319309237)
    assert np.std(fixations_train.t) == approx(5.2387518223254474)
    assert np.std(fixations_train.lengths) == approx(0.0)

    assert kurtosis(fixations_train.x) == approx(-0.6327397503173677)
    assert kurtosis(fixations_train.y) == approx(-0.3662318210834883)
    assert kurtosis(fixations_train.t) == approx(5.6123414320267795)
    assert kurtosis(fixations_train.lengths) == approx(-3.0)

    assert skew(fixations_train.x) == approx(0.10139095797827476)
    assert skew(fixations_train.y) == approx(0.13853441448148346)
    assert skew(fixations_train.t) == approx(1.8891615714930796)
    assert skew(fixations_train.lengths) == approx(0.0)

    assert entropy(fixations_train.n) == approx(13.22601241838667)
    assert (fixations_train.n == 0).sum() == 170


    assert len(fixations_val.x) == 1662655

    assert np.mean(fixations_val.x) == approx(308.64650213062845)
    assert np.mean(fixations_val.y) == approx(217.97772358065865)
    assert np.mean(fixations_val.t) == approx(4.808886389539622)
    assert np.mean(fixations_val.lengths) == approx(0.0)

    assert np.std(fixations_val.x) == approx(130.34460214133043)
    assert np.std(fixations_val.y) == approx(85.80831530782285)
    assert np.std(fixations_val.t) == approx(4.999870176048051)
    assert np.std(fixations_val.lengths) == approx(0.0)

    assert kurtosis(fixations_val.x) == approx(-0.5958648294721907)
    assert kurtosis(fixations_val.y) == approx(-0.31300073559578934)
    assert kurtosis(fixations_val.t) == approx(4.9489750451359225)
    assert kurtosis(fixations_val.lengths) == approx(-3.0)

    assert skew(fixations_val.x) == approx(0.11714467225615313)
    assert skew(fixations_val.y) == approx(0.12631245881037118)
    assert skew(fixations_val.t) == approx(1.8301317514860862)
    assert skew(fixations_val.lengths) == approx(0.0)

    assert entropy(fixations_val.n) == approx(12.234936723301066)
    assert (fixations_val.n == 0).sum() == 259

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2017_mouse(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2017', fixation_type='mouse')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()

    assert len(fixations_train.x) == 215286274

    assert np.mean(fixations_train.x) == approx(314.91750797871686)
    assert np.mean(fixations_train.y) == approx(232.38085973332957)
    assert np.mean(fixations_train.t) == approx(2541.6537073654777)
    assert np.mean(fixations_train.lengths) == approx(0.0)

    assert np.std(fixations_train.x) == approx(138.09403491170718)
    assert np.std(fixations_train.y) == approx(93.55417139372516)
    assert np.std(fixations_train.t) == approx(1432.604664553447)
    assert np.std(fixations_train.lengths) == approx(0.0)

    assert kurtosis(fixations_train.x) == approx(-0.8009690077811422)
    assert kurtosis(fixations_train.y) == approx(-0.638316482844866)
    assert kurtosis(fixations_train.t) == approx(6854.681620924244)
    assert kurtosis(fixations_train.lengths) == approx(-3.0)

    assert skew(fixations_train.x) == approx(0.06734542626655958)
    assert skew(fixations_train.y) == approx(0.07252065918701057)
    assert skew(fixations_train.t) == approx(17.770454294178407)
    assert skew(fixations_train.lengths) == approx(0.0)

    assert entropy(fixations_train.n) == approx(13.274472019581758)
    assert (fixations_train.n == 0).sum() == 24496


    assert len(fixations_val.x) == 121898426

    assert np.mean(fixations_val.x) == approx(313.3112383249313)
    assert np.mean(fixations_val.y) == approx(231.8708303160281)
    assert np.mean(fixations_val.t) == approx(2538.2123597970003)
    assert np.mean(fixations_val.lengths) == approx(0.0)

    assert np.std(fixations_val.x) == approx(139.30115624028937)
    assert np.std(fixations_val.y) == approx(95.24435516821612)
    assert np.std(fixations_val.t) == approx(1395.986706164002)
    assert np.std(fixations_val.lengths) == approx(0.0)

    assert kurtosis(fixations_val.x) == approx(-0.7932049483979013)
    assert kurtosis(fixations_val.y) == approx(-0.6316552996345393)
    assert kurtosis(fixations_val.t) == approx(-1.1483055562729023)
    assert kurtosis(fixations_val.lengths) == approx(-3.0)

    assert skew(fixations_val.x) == approx(0.08023882420460927)
    assert skew(fixations_val.y) == approx(0.07703227629250083)
    assert skew(fixations_val.t) == approx(-0.0027158508337847653)
    assert skew(fixations_val.lengths) == approx(0.0)

    assert entropy(fixations_val.n) == approx(12.278275960422771)
    assert (fixations_val.n == 0).sum() == 23961

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2017_fixations(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2017', fixation_type='fixations')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()

    assert len(fixations_train.x) == 4598112

    assert np.mean(fixations_train.x) == approx(314.62724265959594)
    assert np.mean(fixations_train.y) == approx(228.43566163677613)
    assert np.mean(fixations_train.t) == approx(4.692611228260643)
    assert np.mean(fixations_train.lengths) == approx(0.0)

    assert np.std(fixations_train.x) == approx(134.1455759990284)
    assert np.std(fixations_train.y) == approx(87.13212105359052)
    assert np.std(fixations_train.t) == approx(3.7300713016372375)
    assert np.std(fixations_train.lengths) == approx(0.0)

    assert kurtosis(fixations_train.x) == approx(-0.8163385970402013)
    assert kurtosis(fixations_train.y) == approx(-0.615440115290188)
    assert kurtosis(fixations_train.t) == approx(0.7328902767227148)
    assert kurtosis(fixations_train.lengths) == approx(-3.0)

    assert skew(fixations_train.x) == approx(0.07523280051849487)
    assert skew(fixations_train.y) == approx(0.0854479359829959)
    assert skew(fixations_train.t) == approx(0.8951438604006022)
    assert skew(fixations_train.lengths) == approx(0.0)

    assert entropy(fixations_train.n) == approx(13.26103635730998)
    assert (fixations_train.n == 0).sum() == 532


    assert len(fixations_val.x) == 2576914

    assert np.mean(fixations_val.x) == approx(312.8488630198757)
    assert np.mean(fixations_val.y) == approx(227.6883237081253)
    assert np.mean(fixations_val.t) == approx(4.889936955598829)
    assert np.mean(fixations_val.lengths) == approx(0.0)

    assert np.std(fixations_val.x) == approx(133.22242352479964)
    assert np.std(fixations_val.y) == approx(86.71553440419093)
    assert np.std(fixations_val.t) == approx(3.9029124873868466)
    assert np.std(fixations_val.lengths) == approx(0.0)

    assert kurtosis(fixations_val.x) == approx(-0.7961636859307624)
    assert kurtosis(fixations_val.y) == approx(-0.5897615692354612)
    assert kurtosis(fixations_val.t) == approx(0.7766482713546012)
    assert kurtosis(fixations_val.lengths) == approx(-3.0)

    assert skew(fixations_val.x) == approx(0.08676607299583787)
    assert skew(fixations_val.y) == approx(0.08801482949432776)
    assert skew(fixations_val.t) == approx(0.9082922185416067)
    assert skew(fixations_val.lengths) == approx(0.0)

    assert entropy(fixations_val.n) == approx(12.259608288646687)
    assert (fixations_val.n == 0).sum() == 593

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)

@pytest.mark.slow
@pytest.mark.download
def test_PASCAL_S(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_PASCAL_S(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('PASCAL-S/stimuli.hdf5').check()
        assert location.join('PASCAL-S/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 850

    assert len(fixations.x) == 40314

    assert np.mean(fixations.x) == approx(240.72756362553952)
    assert np.mean(fixations.y) == approx(194.85756809048965)
    assert np.mean(fixations.t) == approx(2.7856823932132757)
    assert np.mean(fixations.lengths) == approx(2.7856823932132757)

    assert np.std(fixations.x) == approx(79.57401169717699)
    assert np.std(fixations.y) == approx(65.21296890260112)
    assert np.std(fixations.t) == approx(2.1191752645988675)
    assert np.std(fixations.lengths) == approx(2.1191752645988675)

    assert kurtosis(fixations.x) == approx(0.0009226786675387011)
    assert kurtosis(fixations.y) == approx(1.1907544566979986)
    assert kurtosis(fixations.t) == approx(-0.540943536495714)
    assert kurtosis(fixations.lengths) == approx(-0.540943536495714)

    assert skew(fixations.x) == approx(0.2112334873314548)
    assert skew(fixations.y) == approx(0.7208733522533084)
    assert skew(fixations.t) == approx(0.4800678710338635)
    assert skew(fixations.lengths) == approx(0.4800678710338635)

    assert entropy(fixations.n) == approx(9.711222735065062)
    assert (fixations.n == 0).sum() == 35

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
def test_DUT_OMRON(location, tmpdir):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_DUT_OMRON(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('DUT-OMRON/stimuli.hdf5').check()
        assert location.join('DUT-OMRON/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 5168

    assert len(fixations.x) == 797542

    assert np.mean(fixations.x) == approx(182.16198519952553)
    assert np.mean(fixations.y) == approx(147.622566585835)
    assert np.mean(fixations.t) == approx(21.965026293286122)
    assert np.mean(fixations.lengths) == approx(21.965026293286122)

    assert np.std(fixations.x) == approx(64.01040053828082)
    assert np.std(fixations.y) == approx(58.292098903584176)
    assert np.std(fixations.t) == approx(17.469479262739807)
    assert np.std(fixations.lengths) == approx(17.469479262739807)

    assert kurtosis(fixations.x) == approx(-0.0689271960358524)
    assert kurtosis(fixations.y) == approx(0.637871926687533)
    assert kurtosis(fixations.t) == approx(2.914601085582113)
    assert kurtosis(fixations.lengths) == approx(2.914601085582113)

    assert skew(fixations.x) == approx(0.23776167825897998)
    assert skew(fixations.y) == approx(0.6328497077003701)
    assert skew(fixations.t) == approx(1.2911168563657345)
    assert skew(fixations.lengths) == approx(1.2911168563657345)

    assert entropy(fixations.n) == approx(12.20642017670851)
    assert (fixations.n == 0).sum() == 209

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.nonfree
@pytest.mark.skip_octave
def test_koehler(location):
    real_location = _location(location)

    stimuli, fixations_freeviewing, fixations_objectsearch, fixations_saliencysearch \
        = pysaliency.external_datasets.get_koehler(location=real_location, datafile='ThirdParty/Koehler_PublicData.zip')

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('Koehler/stimuli.hdf5').check()
        assert location.join('Koehler/fixations_freeviewing.hdf5').check()
        assert location.join('Koehler/fixations_objectsearch.hdf5').check()
        assert location.join('Koehler/fixations_saliencysearch.hdf5').check()

    assert len(stimuli.stimuli) == 800
    assert set(stimuli.sizes) == {(405, 405)}

    assert len(fixations_freeviewing.x) == 94600

    assert np.mean(fixations_freeviewing.x) == approx(205.99696617336153)
    assert np.mean(fixations_freeviewing.y) == approx(190.69461945031713)
    assert np.mean(fixations_freeviewing.t) == approx(2.6399788583509514)
    assert np.mean(fixations_freeviewing.lengths) == approx(2.6399788583509514)

    assert np.std(fixations_freeviewing.x) == approx(86.7891146642233)
    assert np.std(fixations_freeviewing.y) == approx(67.11495414894833)
    assert np.std(fixations_freeviewing.t) == approx(2.1333811216291982)
    assert np.std(fixations_freeviewing.lengths) == approx(2.1333811216291982)

    assert kurtosis(fixations_freeviewing.x) == approx(-0.6927977738542421)
    assert kurtosis(fixations_freeviewing.y) == approx(0.26434562598200007)
    assert kurtosis(fixations_freeviewing.t) == approx(1.0000780305443921)
    assert kurtosis(fixations_freeviewing.lengths) == approx(1.0000780305443921)

    assert skew(fixations_freeviewing.x) == approx(0.04283261395632401)
    assert skew(fixations_freeviewing.y) == approx(0.15277972804817913)
    assert skew(fixations_freeviewing.t) == approx(0.8569222723327634)
    assert skew(fixations_freeviewing.lengths) == approx(0.8569222723327634)

    assert entropy(fixations_freeviewing.n) == approx(9.638058218898772)
    assert (fixations_freeviewing.n == 0).sum() == 128

    assert len(fixations_objectsearch.x) == 125293

    assert np.mean(fixations_objectsearch.x) == approx(199.05052955871437)
    assert np.mean(fixations_objectsearch.y) == approx(202.8867534499134)
    assert np.mean(fixations_objectsearch.t) == approx(3.9734302794250276)
    assert np.mean(fixations_objectsearch.lengths) == approx(3.9734302794250276)

    assert np.std(fixations_objectsearch.x) == approx(88.10778886056328)
    assert np.std(fixations_objectsearch.y) == approx(65.29208873896408)
    assert np.std(fixations_objectsearch.t) == approx(2.902206977368411)
    assert np.std(fixations_objectsearch.lengths) == approx(2.902206977368411)

    assert kurtosis(fixations_objectsearch.x) == approx(-0.49120093084140537)
    assert kurtosis(fixations_objectsearch.y) == approx(0.625841808353278)
    assert kurtosis(fixations_objectsearch.t) == approx(-0.33967380087822274)
    assert kurtosis(fixations_objectsearch.lengths) == approx(-0.33967380087822274)

    assert skew(fixations_objectsearch.x) == approx(0.12557741560793217)
    assert skew(fixations_objectsearch.y) == approx(-0.005003252610602025)
    assert skew(fixations_objectsearch.t) == approx(0.5297789219605314)
    assert skew(fixations_objectsearch.lengths) == approx(0.5297789219605314)

    assert entropy(fixations_objectsearch.n) == approx(9.637156128022387)
    assert (fixations_objectsearch.n == 0).sum() == 140

    assert len(fixations_saliencysearch.x) == 94528

    assert np.mean(fixations_saliencysearch.x) == approx(203.7605894549763)
    assert np.mean(fixations_saliencysearch.y) == approx(193.67308099187542)
    assert np.mean(fixations_saliencysearch.t) == approx(2.7536814488828707)
    assert np.mean(fixations_saliencysearch.lengths) == approx(2.7536814488828707)

    assert np.std(fixations_saliencysearch.x) == approx(94.18304559956722)
    assert np.std(fixations_saliencysearch.y) == approx(65.3335501279418)
    assert np.std(fixations_saliencysearch.t) == approx(2.114709138575087)
    assert np.std(fixations_saliencysearch.lengths) == approx(2.114709138575087)

    assert kurtosis(fixations_saliencysearch.x) == approx(-0.9085078389136778)
    assert kurtosis(fixations_saliencysearch.y) == approx(0.319385892621745)
    assert kurtosis(fixations_saliencysearch.t) == approx(-0.06720050297739633)
    assert kurtosis(fixations_saliencysearch.lengths) == approx(-0.06720050297739633)

    assert skew(fixations_saliencysearch.x) == approx(0.0019227173784863957)
    assert skew(fixations_saliencysearch.y) == approx(0.05728474858602427)
    assert skew(fixations_saliencysearch.t) == approx(0.5866228411986677)
    assert skew(fixations_saliencysearch.lengths) == approx(0.5866228411986677)

    assert entropy(fixations_saliencysearch.n) == approx(9.639365034197382)
    assert (fixations_saliencysearch.n == 0).sum() == 103

    assert len(fixations_freeviewing) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_freeviewing))
    assert len(fixations_objectsearch) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_objectsearch))
    assert len(fixations_saliencysearch) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_saliencysearch))


@pytest.mark.slow
@pytest.mark.download
def test_FIGRIM(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_FIGRIM(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('FIGRIM/stimuli.hdf5').check()
        assert location.join('FIGRIM/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 2793
    assert set(stimuli.sizes) == {(1000, 1000)}

    assert len(fixations.x) == 424712

    assert np.mean(fixations.x) == approx(504.64437312814323)
    assert np.mean(fixations.y) == approx(512.8821982896645)
    assert np.mean(fixations.t) == approx(2.9365758443368684)
    assert np.mean(fixations.lengths) == approx(2.9365758443368684)

    assert np.std(fixations.x) == approx(158.86601835411133)
    assert np.std(fixations.y) == approx(145.67212772412645)
    assert np.std(fixations.t) == approx(2.1599063813289363)
    assert np.std(fixations.lengths) == approx(2.1599063813289363)

    assert kurtosis(fixations.x) == approx(0.5791564709307742)
    assert kurtosis(fixations.y) == approx(0.709663215799134)
    assert kurtosis(fixations.t) == approx(-0.7245566668044039)
    assert kurtosis(fixations.lengths) == approx(-0.7245566668044039)

    assert skew(fixations.x) == approx(0.09245444798073615)
    assert skew(fixations.y) == approx(-0.008328881229649684)
    assert skew(fixations.t) == approx(0.37950203945703337)
    assert skew(fixations.lengths) == approx(0.37950203945703337)

    assert (fixations.n == 0).sum() == 107

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
def test_OSIE(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_OSIE(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('OSIE/stimuli.hdf5').check()
        assert location.join('OSIE/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 700
    assert set(stimuli.sizes) == {(600, 800)}

    assert len(fixations.x) == 98321

    assert np.mean(fixations.x) == approx(401.466024552232)
    assert np.mean(fixations.y) == approx(283.58293548682377)
    assert np.mean(fixations.t) == approx(4.369971826974909)
    assert np.mean(fixations.lengths) == approx(4.369971826974909)

    assert np.std(fixations.x) == approx(171.2760014573171)
    assert np.std(fixations.y) == approx(117.70943331958269)
    assert np.std(fixations.t) == approx(2.9882005810119465)
    assert np.std(fixations.lengths) == approx(2.9882005810119465)

    assert kurtosis(fixations.x) == approx(-0.8437559648105699)
    assert kurtosis(fixations.y) == approx(-0.6146702485058717)
    assert kurtosis(fixations.t) == approx(-0.7170874454173752)
    assert kurtosis(fixations.lengths) == approx(-0.7170874454173752)

    assert skew(fixations.x) == approx(0.029428873077089763)
    assert skew(fixations.y) == approx(0.170931952813165)
    assert skew(fixations.t) == approx(0.31008498461792156)
    assert skew(fixations.lengths) == approx(0.31008498461792156)

    assert entropy(fixations.n) == approx(9.445962418853439)
    assert (fixations.n == 0).sum() == 141

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
def test_NUSEF(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_NUSEF_public(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('NUSEF_public/stimuli.hdf5').check()
        assert location.join('NUSEF_public/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 444

    assert len(fixations.x) == 71477

    assert np.mean(fixations.x) == approx(515.7081586714573)
    assert np.mean(fixations.y) == approx(339.39582588745634)
    assert np.mean(fixations.t) == approx(9.704377576003472)
    assert np.mean(fixations.lengths) == approx(4.205604600081145)

    assert np.std(fixations.x) == approx(164.706599106392)
    assert np.std(fixations.y) == approx(138.0916655852643)
    assert np.std(fixations.t) == approx(15.045168261403202)
    assert np.std(fixations.lengths) == approx(3.5120574118479087)

    assert kurtosis(fixations.x) == approx(1.0061621405730756)
    assert kurtosis(fixations.y) == approx(1.324567134330601)
    assert kurtosis(fixations.t) == approx(2.378559181643473)
    assert kurtosis(fixations.lengths) == approx(0.7152102743878705)

    assert skew(fixations.x) == approx(0.1128294554690726)
    assert skew(fixations.y) == approx(0.5176640896547959)
    assert skew(fixations.t) == approx(1.9080635569791038)
    assert skew(fixations.lengths) == approx(0.9617232401848489)

    assert (fixations.n == 0).sum() == 132

    # not testing this, there are many out-of-stimulus fixations in the dataset
    # assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))



@pytest.mark.slow
@pytest.mark.download
def test_COCO_Search18(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_val, fixations_val = pysaliency.external_datasets.get_COCO_Search18(location=real_location)
    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.Stimuli)
        assert not isinstance(stimuli_val, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.FileStimuli)
        assert location.join('COCO-Search18/stimuli_train.hdf5').check()
        assert location.join('COCO-Search18/stimuli_validation.hdf5').check()
        assert location.join('COCO-Search18/fixations_train.hdf5').check()
        assert location.join('COCO-Search18/fixations_validation.hdf5').check()

    assert len(stimuli_train) == 3714
    assert len(stimuli_val) == 623
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}

    assert len(fixations_train.x) == 207970

    assert np.mean(fixations_train.x) == approx(835.8440337548686)
    assert np.mean(fixations_train.y) == approx(509.6030908304083)
    assert np.mean(fixations_train.t) == approx(3.0987979035437805)
    assert np.mean(fixations_train.lengths) == approx(3.0987979035437805)

    assert np.std(fixations_train.x) == approx(336.5760343388881)
    assert np.std(fixations_train.y) == approx(193.04654731407436)
    assert np.std(fixations_train.t) == approx(3.8411822348178664)
    assert np.std(fixations_train.lengths) == approx(3.8411822348178664)

    assert kurtosis(fixations_train.x) == approx(-0.6283401149747818)
    assert kurtosis(fixations_train.y) == approx(0.15947671647330974)
    assert kurtosis(fixations_train.t) == approx(12.038491881119654)
    assert kurtosis(fixations_train.lengths) == approx(12.038491881119654)

    assert skew(fixations_train.x) == approx(0.1706207784149093)
    assert skew(fixations_train.y) == approx(-0.07268825958515616)
    assert skew(fixations_train.t) == approx(2.804671690266736)
    assert skew(fixations_train.lengths) == approx(2.804671690266736)

    assert entropy(fixations_train.n) == approx(11.654309812153487)
    assert (fixations_train.n == 0).sum() == 48

    assert len(fixations_val.x) == 31761

    assert np.mean(fixations_val.x) == approx(841.0752652624287)
    assert np.mean(fixations_val.y) == approx(498.3305594911999)
    assert np.mean(fixations_val.t) == approx(3.107994080790907)
    assert np.mean(fixations_val.lengths) == approx(3.107994080790907)

    assert np.std(fixations_val.x) == approx(331.6328528765362)
    assert np.std(fixations_val.y) == approx(195.86110035077112)
    assert np.std(fixations_val.t) == approx(3.7502120687824454)
    assert np.std(fixations_val.lengths) == approx(3.7502120687824454)

    assert kurtosis(fixations_val.x) == approx(-0.5973130907561486)
    assert kurtosis(fixations_val.y) == approx(-0.2797786304225598)
    assert kurtosis(fixations_val.t) == approx(11.250011182161305)
    assert kurtosis(fixations_val.lengths) == approx(11.250011182161305)

    assert skew(fixations_val.x) == approx(0.14886675209256964)
    assert skew(fixations_val.y) == approx(-0.04086275403802345)
    assert skew(fixations_val.t) == approx(2.671653646130074)
    assert skew(fixations_val.lengths) == approx(2.671653646130074)

    assert entropy(fixations_val.n) == approx(9.159600084079305)
    assert (fixations_val.n == 0).sum() == 52

    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))
