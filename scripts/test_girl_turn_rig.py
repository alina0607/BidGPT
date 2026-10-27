"""Geometric/source-integrity checks; these do not grant visual approval."""
import unittest
import numpy as np
from PIL import Image
from build_girl_turn_rig import (
    SIZE, OUT, source_frame, mesh_data, render, signed_area, HANDLES,
)


class OriginalPixelRigTests(unittest.TestCase):
    def test_rest_is_original(self):
        idle=source_frame('girl_idle',0)
        source,delta,topology=mesh_data()
        result,_=render(source,delta,topology,idle,0)
        self.assertEqual(result.tobytes(),idle.tobytes())

    def test_multirow_source_extraction(self):
        # The original walk is a 16-column TWO-row atlas, not 30 columns.
        frame=source_frame('girl_walk',17)
        self.assertGreater(np.count_nonzero(np.array(frame)[:,:,3]),20000)
        self.assertNotEqual(frame.tobytes(),source_frame('girl_walk',15).tobytes())

    def test_layers_cover_each_source_pixel_once(self):
        original=np.array(source_frame('girl_idle',0))
        layers=[np.array(Image.open(p)) for p in (OUT/'layers').glob('*.png')]
        self.assertEqual(len(layers),8)
        self.assertTrue(np.array_equal(np.sum(layers,axis=0),original))

    def test_whole_small_angle_range_has_no_foldovers(self):
        source,delta,topology=mesh_data()
        rest=signed_area(source,topology)
        for p in np.linspace(0,1,121):
            self.assertGreater((signed_area(source+delta*p,topology)/rest).min(),.7)

    def test_no_foot_or_crown_scaling(self):
        handles={n:(p,d) for n,p,d in HANDLES}
        for name in ('crown','cap_l','cap_r','cap_top_l','cap_top_r',
                     'ankle_near','ankle_far','sole_near','sole_far'):
            self.assertEqual(handles[name][1],(0,0),name)

    def test_alpha_sampling_has_no_white_or_black_fringe(self):
        # A constant coloured texture with transparent exterior must remain that
        # same colour at its antialiased silhouette after mesh sampling.
        pixels=np.zeros((SIZE[1],SIZE[0],4),np.uint8)
        pixels[120:300,120:300]=[237,120,170,255]
        src,delta,topology=mesh_data()
        frame,stats=render(src,delta,topology,Image.fromarray(pixels),.53)
        result=np.array(frame)
        visible=result[:,:,3]>0
        self.assertTrue(np.all(result[:,:,:3][visible]==[237,120,170]))
        self.assertEqual(stats['uncovered_pixels'],0)


if __name__=='__main__':
    unittest.main()
